package vn.hifpt.clickstream

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max

class MainActivity : Activity() {
    private val handler = Handler(Looper.getMainLooper())
    private val modelExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    private lateinit var statsView: TextView
    private lateinit var sessionSpinner: Spinner
    private lateinit var speedSpinner: Spinner
    private lateinit var startButton: Button
    private lateinit var pauseButton: Button
    private lateinit var stopButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var statusView: TextView
    private lateinit var currentEventView: TextView
    private lateinit var analyzeButton: Button
    private lateinit var modelResultView: TextView
    private lateinit var logView: TextView

    private var sessions: List<ReplaySession> = emptyList()
    private var currentSession: ReplaySession? = null
    private var replayIndex = 0
    private var replayRunning = false
    private var replayRunnable: Runnable? = null
    private val logLines = ArrayDeque<String>()
    private var mobileModel: MobileJourneyModel? = null
    private var liveStream: MobileJourneyStream? = null
    private var liveWindowStartTimestamp: Long? = null
    private var liveWindowIndex = 1
    private val liveFinalized = mutableListOf<MobilePrediction>()
    private val liveGeneration = AtomicInteger(0)
    private var analysisInProgress = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildContent())
        loadData()
        loadMobileModel()
    }

    private fun buildContent(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(16))
            setBackgroundColor(Color.rgb(246, 248, 250))
        }

        root.addView(TextView(this).apply {
            text = "Clickstream Simulator"
            textSize = 24f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.rgb(13, 71, 161))
        }, lp())
        root.addView(TextView(this).apply {
            text = "Replay journey từ test_data_july.json"
            textSize = 14f
            setTextColor(Color.DKGRAY)
            setPadding(0, dp(3), 0, dp(12))
        }, lp())

        statsView = TextView(this).apply {
            text = "Đang tải dữ liệu..."
            textSize = 14f
            setTextColor(Color.rgb(33, 33, 33))
            setPadding(dp(12), dp(10), dp(12), dp(10))
            background = roundedBackground(Color.WHITE, Color.rgb(220, 226, 232))
        }
        root.addView(statsView, lp(bottom = 10))

        root.addView(label("Session replay"), lp())
        sessionSpinner = Spinner(this)
        root.addView(sessionSpinner, lp(bottom = 8))

        root.addView(label("Tốc độ"), lp())
        speedSpinner = Spinner(this)
        root.addView(speedSpinner, lp(bottom = 10))
        speedSpinner.adapter = spinnerAdapter(listOf("0.25x", "1x", "5x", "10x"))
        speedSpinner.setSelection(1)

        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        startButton = button("Start").also { it.isEnabled = false }
        pauseButton = button("Pause").also { it.isEnabled = false }
        stopButton = button("Stop").also { it.isEnabled = false }
        buttons.addView(startButton, weightLp())
        buttons.addView(pauseButton, weightLp())
        buttons.addView(stopButton, weightLp())
        root.addView(buttons, lp(bottom = 10))

        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = 0
        }
        root.addView(progressBar, lp(bottom = 5))

        statusView = TextView(this).apply {
            text = "Chưa sẵn sàng"
            textSize = 13f
            setTextColor(Color.DKGRAY)
            setPadding(0, 0, 0, dp(8))
        }
        root.addView(statusView, lp())

        currentEventView = TextView(this).apply {
            text = "Event hiện tại sẽ hiển thị ở đây"
            textSize = 15f
            typeface = Typeface.MONOSPACE
            setTextColor(Color.rgb(20, 45, 70))
            setPadding(dp(12), dp(12), dp(12), dp(12))
            background = roundedBackground(Color.WHITE, Color.rgb(187, 208, 230))
        }
        root.addView(currentEventView, lp(bottom = 10))

        val modelRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        analyzeButton = button("Analyze 30s windows").also { it.isEnabled = false }
        modelRow.addView(analyzeButton, LinearLayout.LayoutParams(-1, -2))
        root.addView(modelRow, lp(bottom = 6))
        modelResultView = TextView(this).apply {
            text = "Model output theo từng 30s window sẽ hiển thị ở đây"
            textSize = 12f
            typeface = Typeface.MONOSPACE
            setTextColor(Color.rgb(20, 65, 45))
            setPadding(dp(10), dp(10), dp(10), dp(10))
            background = roundedBackground(Color.rgb(247, 253, 249), Color.rgb(176, 213, 190))
        }
        root.addView(modelResultView, lp(bottom = 10))

        root.addView(label("Event log"), lp())
        logView = TextView(this).apply {
            text = ""
            textSize = 12f
            typeface = Typeface.MONOSPACE
            setTextColor(Color.rgb(40, 40, 40))
            setPadding(dp(10), dp(10), dp(10), dp(10))
            background = roundedBackground(Color.rgb(250, 250, 250), Color.rgb(220, 226, 232))
        }
        val logScroll = ScrollView(this).apply {
            addView(logView, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(logScroll, LinearLayout.LayoutParams(-1, 0, 1f))

        sessionSpinner.onItemSelectedListener = SimpleItemSelectedListener { position ->
            if (position in sessions.indices) {
                if (replayRunning) stopReplay(reset = true)
                currentSession = sessions[position]
                replayIndex = 0
                clearReplayDisplay()
                updateSessionStatus()
            }
        }
        startButton.setOnClickListener { startReplay() }
        pauseButton.setOnClickListener { pauseReplay() }
        stopButton.setOnClickListener { stopReplay(reset = true) }
        analyzeButton.setOnClickListener { analyzeCurrentSession() }

        return root
    }

    private fun loadData() {
        Thread {
            try {
                val loaded = ClickstreamRepository(this).loadSessions()
                runOnUiThread { onDataLoaded(loaded) }
            } catch (error: Exception) {
                runOnUiThread {
                    statusView.text = "Không đọc được asset: ${error.message}"
                    statsView.text = "Kiểm tra app/src/main/assets/test_data_july.json"
                }
            }
        }.start()
    }

    private fun loadMobileModel() {
        Thread {
            try {
                val loaded = MobileJourneyModel(this)
                runOnUiThread {
                    mobileModel = loaded
                    analyzeButton.isEnabled = currentSession != null
            modelResultView.text = "Model Android đã sẵn sàng · ONNX + Markov · window 30s"
                }
            } catch (error: Exception) {
                runOnUiThread {
                    modelResultView.text = "Không load được model: ${error.message}"
                }
            }
        }.start()
    }

    private fun onDataLoaded(loaded: List<ReplaySession>) {
        sessions = loaded
        currentSession = sessions.firstOrNull()
        val eventCount = sessions.sumOf { it.events.size }
        val platforms = sessions.flatMap { it.events }.map { it.segment }
            .filter { it.isNotBlank() }.groupingBy { it }.eachCount()
        statsView.text = "${formatNumber(eventCount)} events  ·  ${sessions.size} sessions\n" +
            platforms.entries.joinToString("  ·  ") { "${it.key}: ${formatNumber(it.value)}" }

        sessionSpinner.adapter = spinnerAdapter(sessions.mapIndexed { index, session ->
            "${index + 1}. ${session.id.take(8)}…  (${session.events.size} events, ${session.platform})"
        })
        startButton.isEnabled = sessions.isNotEmpty()
        pauseButton.isEnabled = false
        stopButton.isEnabled = sessions.isNotEmpty()
        analyzeButton.isEnabled = sessions.isNotEmpty() && mobileModel != null
        updateSessionStatus()
    }

    private fun startReplay() {
        val session = currentSession ?: return
        if (replayIndex >= session.events.size) {
            replayIndex = 0
            clearReplayDisplay()
        }
        if (replayIndex == 0) prepareLiveAnalysis()
        replayRunning = true
        startButton.isEnabled = false
        pauseButton.isEnabled = true
        stopButton.isEnabled = true
        statusView.text = "Đang replay ${session.id} · event ${replayIndex + 1}/${session.events.size}"
        scheduleNext()
    }

    private fun scheduleNext() {
        val session = currentSession ?: return
        if (!replayRunning || replayIndex >= session.events.size) {
            finishReplay()
            return
        }

        val delay = if (replayIndex == 0) {
            0L
        } else {
            val previous = session.events[replayIndex - 1]
            val current = session.events[replayIndex]
            val speed = speedMultiplier()
            max(40L, ((current.timestamp - previous.timestamp).coerceAtLeast(0L) / speed).toLong())
        }
        replayRunnable = Runnable {
            if (!replayRunning) return@Runnable
            emit(session.events[replayIndex])
            replayIndex += 1
            scheduleNext()
        }.also { handler.postDelayed(it, delay) }
    }

    private fun emit(event: ClickstreamEvent) {
        val session = currentSession ?: return
        val progress = if (session.events.isEmpty()) 0 else (replayIndex * 100 / session.events.size)
        progressBar.progress = progress
        currentEventView.text = buildString {
            append(event.key.uppercase(Locale.US))
            append("  ")
            append(event.name)
            append("\n")
            append("time: ")
            append(dateFormat.format(Date(event.timestamp)))
            append("  |  platform: ")
            append(event.segment.ifBlank { "-" })
            append("\ndevice: ")
            append(event.deviceId)
            append("  |  customer: ")
            append(event.customerId ?: "anonymous")
            if (event.screenId.isNotBlank()) append("\nscreen: ${event.screenId}")
        }
        addLog("${dateFormat.format(Date(event.timestamp))}  ${event.key.padEnd(10)}  ${event.name}")
        statusView.text = "Đang replay ${session.id} · event ${replayIndex + 1}/${session.events.size}"
        enqueueLiveEvent(event)
    }

    private fun pauseReplay() {
        replayRunning = false
        replayRunnable?.let(handler::removeCallbacks)
        replayRunnable = null
        pauseButton.isEnabled = false
        startButton.isEnabled = currentSession != null
        statusView.text = "Đã pause tại event $replayIndex"
    }

    private fun stopReplay(reset: Boolean) {
        replayRunning = false
        replayRunnable?.let(handler::removeCallbacks)
        replayRunnable = null
        if (reset) {
            liveGeneration.incrementAndGet()
            replayIndex = 0
            clearReplayDisplay()
            updateSessionStatus()
        }
        pauseButton.isEnabled = false
        startButton.isEnabled = currentSession != null
    }

    private fun finishReplay() {
        replayRunning = false
        replayRunnable = null
        pauseButton.isEnabled = false
        startButton.isEnabled = currentSession != null
        progressBar.progress = 100
        statusView.text = "Hoàn tất replay ${currentSession?.id ?: ""}"
        finishLiveAnalysis()
    }

    private fun clearReplayDisplay() {
        progressBar.progress = 0
        currentEventView.text = "Event hiện tại sẽ hiển thị ở đây"
        modelResultView.text = if (mobileModel == null) {
            "Model đang tải hoặc chưa có trong assets"
        } else {
            "Model output theo từng 30s window sẽ hiển thị ở đây"
        }
        logLines.clear()
        logView.text = ""
    }

    private fun prepareLiveAnalysis() {
        val model = mobileModel ?: run {
            liveStream = null
            return
        }
        liveStream = MobileJourneyStream(model)
        liveGeneration.incrementAndGet()
        liveWindowStartTimestamp = null
        liveWindowIndex = 1
        liveFinalized.clear()
        modelResultView.text = "Đang thu event theo transport window 30s..."
    }

    private fun enqueueLiveEvent(event: ClickstreamEvent) {
        val stream = liveStream ?: return
        val generation = liveGeneration.get()
        modelExecutor.execute {
            if (generation != liveGeneration.get()) return@execute
            if (liveWindowStartTimestamp == null) liveWindowStartTimestamp = event.timestamp
            val update = stream.append(event, scoreProvisional = false)
            liveFinalized += update.finalized
            val start = liveWindowStartTimestamp ?: event.timestamp
            if (event.timestamp - start < 30_000L) return@execute

            val window = MobileAnalysisWindow(
                windowIndex = liveWindowIndex++,
                startTimestamp = start,
                endTimestamp = event.timestamp,
                finalized = liveFinalized.toList(),
                provisional = stream.snapshotProvisional()
            )
            liveFinalized.clear()
            liveWindowStartTimestamp = event.timestamp
            runOnUiThread {
                if (generation == liveGeneration.get()) renderLiveWindow(window)
            }
        }
    }

    private fun finishLiveAnalysis() {
        val stream = liveStream
        if (stream == null) {
            analyzeCurrentSession()
            return
        }
        val generation = liveGeneration.get()
        modelExecutor.execute {
            if (generation != liveGeneration.get()) return@execute
            val finalPrediction = stream.flush()
            if (finalPrediction != null) liveFinalized += finalPrediction
            val start = liveWindowStartTimestamp
            if (start != null && liveFinalized.isNotEmpty()) {
                val window = MobileAnalysisWindow(
                    windowIndex = liveWindowIndex,
                    startTimestamp = start,
                    endTimestamp = currentSession?.events?.lastOrNull()?.timestamp ?: start,
                    finalized = liveFinalized.toList(),
                    provisional = null
                )
                liveFinalized.clear()
                runOnUiThread {
                    if (generation == liveGeneration.get()) renderLiveWindow(window)
                }
            }
        }
    }

    private fun updateSessionStatus() {
        val session = currentSession
        statusView.text = if (session == null) "Chưa có session" else {
            "Sẵn sàng · ${session.events.size} events · device ${session.deviceId} · ${session.platform}"
        }
    }

    private fun analyzeCurrentSession() {
        val model = mobileModel ?: run {
            modelResultView.text = "Model đang tải hoặc chưa có trong assets"
            return
        }
        val session = currentSession ?: return
        if (analysisInProgress) return
        analysisInProgress = true
        analyzeButton.isEnabled = false
        modelResultView.text = "Đang chạy preprocessing + ONNX + Markov trên ${session.events.size} events..."
        Thread {
            try {
                val result = model.analyzeEventsInWindows(session.events)
                runOnUiThread {
                    renderWindowedModelResult(result)
                    analysisInProgress = false
                    analyzeButton.isEnabled = true
                }
            } catch (error: Exception) {
                runOnUiThread {
                    modelResultView.text = "Model error: ${error.message}"
                    analysisInProgress = false
                    analyzeButton.isEnabled = true
                }
            }
        }.start()
    }

    private fun renderWindowedModelResult(result: WindowedAnalysisResult) {
        modelResultView.text = buildString {
            append("MODEL OUTPUT · ${result.windows.size} windows × 30s\n")
            if (result.windows.isEmpty()) {
                append("Không có event")
            } else {
                result.windows.takeLast(3).forEach { window ->
                    append("W${window.windowIndex}  finalized=${window.finalized.size}")
                    append("  provisional=${window.provisional?.state ?: "-"}\n")
                    (window.finalized + listOfNotNull(window.provisional)).takeLast(3).forEach { prediction ->
                        append("  ${prediction.journeyId}  cluster=${prediction.cluster ?: "-"}  ${prediction.classCode ?: prediction.state}\n")
                        append("    flags=${prediction.frictionFlags.ifBlank { "-" }}\n")
                        append("    next=${prediction.nextAction ?: "-"}\n")
                    }
                }
            }
        }
    }

    private fun renderLiveWindow(window: MobileAnalysisWindow) {
        modelResultView.text = buildString {
            append("LIVE MODEL OUTPUT · W${window.windowIndex} · 30s\n")
            append("finalized=${window.finalized.size} · provisional=${window.provisional?.state ?: "-"}\n")
            (window.finalized + listOfNotNull(window.provisional)).forEach { prediction ->
                append("${prediction.journeyId}  cluster=${prediction.cluster ?: "-"}  ${prediction.classCode ?: prediction.state}\n")
                append("  flags=${prediction.frictionFlags.ifBlank { "-" }}\n")
                append("  next=${prediction.nextAction ?: "-"}\n")
            }
        }
    }

    private fun addLog(line: String) {
        if (logLines.size >= 120) logLines.removeFirst()
        logLines.addLast(line)
        logView.text = logLines.joinToString("\n")
    }

    private fun speedMultiplier(): Double = when (speedSpinner.selectedItemPosition) {
        0 -> 0.25
        2 -> 5.0
        3 -> 10.0
        else -> 1.0
    }

    private fun spinnerAdapter(values: List<String>) = ArrayAdapter(
        this, android.R.layout.simple_spinner_item, values
    ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }

    private fun button(textValue: String) = Button(this).apply {
        text = textValue
        isAllCaps = false
        minHeight = dp(44)
    }

    private fun label(textValue: String) = TextView(this).apply {
        text = textValue
        textSize = 13f
        typeface = Typeface.DEFAULT_BOLD
        setTextColor(Color.rgb(55, 71, 79))
        setPadding(0, dp(2), 0, dp(2))
    }

    private fun roundedBackground(fill: Int, stroke: Int) = GradientDrawable().apply {
        setColor(fill)
        setStroke(dp(1), stroke)
        cornerRadius = dp(8).toFloat()
    }

    private fun lp(bottom: Int = 0) = LinearLayout.LayoutParams(-1, -2).apply {
        if (bottom > 0) bottomMargin = dp(bottom)
    }

    private fun weightLp() = LinearLayout.LayoutParams(0, -2, 1f).apply {
        marginEnd = dp(4)
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
    private fun formatNumber(value: Int): String = String.format(Locale.US, "%,d", value)

    override fun onDestroy() {
        replayRunnable?.let(handler::removeCallbacks)
        modelExecutor.shutdownNow()
        mobileModel?.close()
        super.onDestroy()
    }
}

private class SimpleItemSelectedListener(private val onSelected: (Int) -> Unit) :
    android.widget.AdapterView.OnItemSelectedListener {
    override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
        onSelected(position)
    }

    override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
}
