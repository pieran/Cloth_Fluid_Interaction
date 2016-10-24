
#include <nclgl\OGLRenderer.h>
#include <nclgl\Window.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\SceneManager.h>
#include <ncltech\NCLDebug.h>
#include <ncltech\..\PerfTimer.h>

#include "FluidScene1.h"
#include "FluidClothScene1.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

const Vector4 status_colour = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
const Vector4 status_colour_header = Vector4(0.8f, 0.9f, 1.0f, 1.0f);

PerfTimer timer_total, timer_physics, timer_update, timer_render;
uint shadowCycleKey = 4;

void Quit(bool error = false, const string &reason = "") {
	//Release Singletons
	SceneManager::Release();
	PhysicsEngine::Release();
	Window::Destroy();

	//Show console reason before exit
	if (error) {
		std::cout << reason << std::endl;
		system("PAUSE");
		exit(-1);
	}
}

bool init_cuda(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}

	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}

	cudaSetDevice(i);
	cudaGLSetGLDevice(i);


	printf("CUDA initialized.\n");
	return true;
}

void Initialize()
{
	//Initialise the Window
	if (!Window::Initialise("Fluid Simulations", 1280, 800, false))
		Quit(true, "Window failed to initialise!");

	//Initialise the PhysicsEngine
	PhysicsEngine::Instance();

	//Initialize Renderer
	SceneManager::Instance()->InitializeOGLContext(Window::GetWindow());
	if (!SceneManager::Instance()->HasInitialised())
		Quit(true, "Renderer failed to initialise!");


	init_cuda();

	//Enqueue All Scenes
	//SceneManager::Instance()->EnqueueScene(new FluidScene1("PICFLUP - Example #1"));
	SceneManager::Instance()->EnqueueScene(new FluidClothScene1("Two Way Coupling - Example #1"));
}

void PrintStatusEntries()
{
	//Print Engine Options
	NCLDebug::AddStatusEntry(status_colour_header, "NCLTech Settings");
	NCLDebug::AddStatusEntry(status_colour, "     Physics Engine: %s (Press P to toggle)", PhysicsEngine::Instance()->IsPaused() ? "Paused  " : "Enabled ");
	NCLDebug::AddStatusEntry(status_colour, "     Monitor V-Sync: %s (Press V to toggle)", SceneManager::Instance()->GetVsyncEnabled() ? "Enabled " : "Disabled");
	NCLDebug::AddStatusEntry(status_colour, "");

	//Print Current Scene Name
	NCLDebug::AddStatusEntry(status_colour_header, "[%d/%d]: %s (Press T/Y to cycle or R to reload)",
		SceneManager::Instance()->GetCurrentSceneIndex() + 1,
		SceneManager::Instance()->SceneCount(),
		SceneManager::Instance()->GetCurrentScene()->GetSceneName().c_str()
		);

	//Print Performance Timers
	NCLDebug::AddStatusEntry(status_colour, "     FPS: %5.2f", 1000.f / timer_total.GetAvg());
	timer_total.PrintOutputToStatusEntry(status_colour, "     Total Time     :");
	timer_update.PrintOutputToStatusEntry(status_colour, "     Scene Update   :");
	timer_physics.PrintOutputToStatusEntry(status_colour, "     Physics Update :");
	timer_render.PrintOutputToStatusEntry(status_colour, "     Render Scene   :");
	NCLDebug::AddStatusEntry(status_colour, "");
}


void HandleKeyboardInputs()
{
	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_P))
		PhysicsEngine::Instance()->SetPaused(!PhysicsEngine::Instance()->IsPaused());

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_V))
		SceneManager::Instance()->SetVsyncEnabled(!SceneManager::Instance()->GetVsyncEnabled());

	uint sceneIdx = SceneManager::Instance()->GetCurrentSceneIndex();
	uint sceneMax = SceneManager::Instance()->SceneCount();
	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_Y))
		SceneManager::Instance()->JumpToScene((sceneIdx + 1) % sceneMax);

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_T))
		SceneManager::Instance()->JumpToScene((sceneIdx == 0 ? sceneMax : sceneIdx) - 1);

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_R))
		SceneManager::Instance()->JumpToScene(sceneIdx);


}


int main()
{
	//Initialize our Window, Physics, Scenes etc
	Initialize();

	Window::GetWindow().GetTimer()->GetTimedMS();

	//Create main game-loop
	while (Window::GetWindow().UpdateWindow() && !Window::GetKeyboard()->KeyDown(KEYBOARD_ESCAPE)){
		//Start Timing
		float dt = Window::GetWindow().GetTimer()->GetTimedMS() * 0.001f;	//How many milliseconds since last update?
		timer_total.BeginTimingSection();

		//Print Status Entries
		PrintStatusEntries();

		//Handle Keyboard Inputs
		HandleKeyboardInputs();

		//Update Performance Timers (Show results every second)
		timer_total.UpdateRealElapsedTime(dt);
		timer_physics.UpdateRealElapsedTime(dt);
		timer_update.UpdateRealElapsedTime(dt);
		timer_render.UpdateRealElapsedTime(dt);

		//Update Scene
		timer_update.BeginTimingSection();
		SceneManager::Instance()->UpdateScene(dt);
		timer_update.EndTimingSection();

		//Update Physics
		timer_physics.BeginTimingSection();
		PhysicsEngine::Instance()->Update(dt);
		timer_physics.EndTimingSection();

		//Render Scene
		timer_render.BeginTimingSection();
		SceneManager::Instance()->RenderScene();
		if (SceneManager::Instance()->GetVsyncEnabled()) glFinish(); //Forces synchronisation if vsync is disabled (Purely for performance timing measurements)
		timer_render.EndTimingSection();

		//Finish Timing
		timer_total.EndTimingSection();

		//Let other programs on the computer have some CPU time
		Sleep(0);
	}

	//Cleanup
	Quit();
	return 0;
}