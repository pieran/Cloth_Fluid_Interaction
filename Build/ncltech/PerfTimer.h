#pragma once
#include <nclgl\GameTimer.h>
#include <ncltech\NCLDebug.h>

class PerfTimer
{
public:
	PerfTimer()
		: m_UpdateInterval(1.0f)
		, m_RealTimeElapsed(0.0f)
	{
		m_Timer.GetTimedMS();
		memset(&m_CurrentData, 0, sizeof(PerfTimer_Data));
		memset(&m_PreviousData, 0, sizeof(PerfTimer_Data));
	}

	virtual ~PerfTimer() {}


	float GetHigh() { return m_PreviousData.maxSample; }
	float GetLow() { return m_PreviousData.minSample; }
	float GetAvg() { return m_PreviousData.sumSamples / float(m_PreviousData.nSamples); }

	void SetUpdateInterval(float seconds) { m_UpdateInterval = seconds; }


	void BeginTimingSection()
	{
		m_Timer.GetTimedMS();

	}

	void EndTimingSection()
	{
		float elapsed = m_Timer.GetTimedMS();

		if (m_CurrentData.nSamples == 0)
		{
			m_CurrentData.maxSample = elapsed;
			m_CurrentData.minSample = elapsed;
		}
		else
		{
			m_CurrentData.maxSample = max(m_CurrentData.maxSample, elapsed);
			m_CurrentData.minSample = min(m_CurrentData.minSample, elapsed);
		}

		m_CurrentData.nSamples++;
		m_CurrentData.sumSamples += elapsed;
	}

	void UpdateRealElapsedTime(float dt)
	{
		m_RealTimeElapsed += dt;
		if (m_RealTimeElapsed >= m_UpdateInterval)
		{
			m_RealTimeElapsed -= m_UpdateInterval;
			m_PreviousData = m_CurrentData;
			memset(&m_CurrentData, 0, sizeof(PerfTimer_Data));
		}
	}

	void PrintOutputToStatusEntry(const Vector4& colour, const std::string& name)
	{
		NCLDebug::AddStatusEntry(colour, "%s%5.2fms [max:%5.2fms, min:%5.2fms]", name.c_str(), GetAvg(), GetHigh(), GetLow());
	}

protected:
	float m_UpdateInterval;
	float m_RealTimeElapsed;

	GameTimer m_Timer;

	struct PerfTimer_Data
	{
		float	maxSample;
		float	minSample;
		float	sumSamples;
		int		nSamples;
	};

	PerfTimer_Data m_PreviousData; //Shown for output
	PerfTimer_Data m_CurrentData; //Still changing
};