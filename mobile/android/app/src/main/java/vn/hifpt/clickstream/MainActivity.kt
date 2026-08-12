package vn.hifpt.clickstream

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
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
import android.widget.EditText
import android.text.InputType
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
    private lateinit var routeSpinner: Spinner
    private lateinit var durationInput: EditText
    private lateinit var startButton: Button
    private lateinit var pauseButton: Button
    private lateinit var stopButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var statusView: TextView
    private lateinit var currentEventView: TextView
    private lateinit var analyzeButton: Button
    private lateinit var modelResultView: TextView
    private lateinit var jsonOutputView: TextView
    private lateinit var copyJsonButton: Button
    private lateinit var logView: TextView
    private var latestJsonOutput = ""

    private var sessions: List<ReplaySession> = emptyList()
    private var currentSession: ReplaySession? = null
    private var replayIndex = 0
    private var replayRunning = false
    private var replayRunnable: Runnable? = null
    private val logLines = ArrayDeque<String>()
    private var mobileModel: MobileJourneyModel? = null
    private var clickstreamProcessor: MobileClickstreamProcessor? = null
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
            text = "Replay journey từ test_data_android.csv"
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

        root.addView(label("Route xử lý clickstream"), lp())
        routeSpinner = Spinner(this)
        routeSpinner.adapter = spinnerAdapter(listOf(
            "Fixed duration · xử lý mỗi khoảng",
            "Journey complete · gom cụm ngay khi hoàn tất"
        ))
        root.addView(routeSpinner, lp(bottom = 8))

        val durationRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        durationRow.addView(label("Duration (giây)"), LinearLayout.LayoutParams(0, -2, 1f))
        durationInput = EditText(this).apply {
            setText("30")
            inputType = InputType.TYPE_CLASS_NUMBER
            hint = "30"
            minWidth = dp(90)
            gravity = Gravity.CENTER
        }
        durationRow.addView(durationInput, LinearLayout.LayoutParams(dp(100), -2))
        root.addView(durationRow, lp(bottom = 10))

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
        analyzeButton = button("Analyze rule-based journeys").also { it.isEnabled = false }
        modelRow.addView(analyzeButton, LinearLayout.LayoutParams(-1, -2))
        root.addView(modelRow, lp(bottom = 6))
        modelResultView = TextView(this).apply {
            text = "Model output sẽ xuất hiện ngay khi rule-based cắt journey"
            textSize = 12f
            typeface = Typeface.MONOSPACE
            setTextColor(Color.rgb(20, 65, 45))
            setPadding(dp(10), dp(10), dp(10), dp(10))
            background = roundedBackground(Color.rgb(247, 253, 249), Color.rgb(176, 213, 190))
        }
        root.addView(modelResultView, lp(bottom = 10))

        val jsonHeader = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        jsonHeader.addView(label("Last JSON output"), LinearLayout.LayoutParams(0, -2, 1f))
        copyJsonButton = button("Copy JSON").also { it.isEnabled = false }
        jsonHeader.addView(copyJsonButton, LinearLayout.LayoutParams(-2, -2))
        root.addView(jsonHeader, lp(bottom = 4))
        jsonOutputView = TextView(this).apply {
            text = "JSON output sẽ xuất hiện sau khi có window hoặc journey hoàn tất"
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setTextColor(Color.rgb(35, 35, 35))
            setPadding(dp(10), dp(10), dp(10), dp(10))
            background = roundedBackground(Color.WHITE, Color.rgb(220, 226, 232))
        }
        val jsonScroll = ScrollView(this).apply {
            addView(jsonOutputView, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(jsonScroll, LinearLayout.LayoutParams(-1, dp(180)).apply {
            bottomMargin = dp(10)
        })

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
        copyJsonButton.setOnClickListener {
            if (latestJsonOutput.isNotBlank()) {
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clipboard.setPrimaryClip(ClipData.newPlainText("clickstream_output.json", latestJsonOutput))
                statusView.text = "Đã copy JSON output"
            }
        }

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
                    statsView.text = "Kiểm tra app/src/main/assets/test_data_android.csv"
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
                    modelResultView.text = "Model Android đã sẵn sàng · boundary-driven inference"
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
            clickstreamProcessor = null
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
            "Chọn route rồi bấm Start để nhận clickstream"
        }
        latestJsonOutput = ""
        jsonOutputView.text = "JSON output sẽ xuất hiện sau khi có window hoặc journey hoàn tất"
        copyJsonButton.isEnabled = false
        logLines.clear()
        logView.text = ""
    }

    private fun prepareLiveAnalysis() {
        val model = mobileModel ?: run {
            clickstreamProcessor = null
            return
        }
        val route = if (routeSpinner.selectedItemPosition == 0) {
            MobileProcessingRoute.FIXED_DURATION
        } else {
            MobileProcessingRoute.JOURNEY_COMPLETE
        }
        val duration = durationInput.text.toString().toLongOrNull()?.coerceIn(1L, 3600L) ?: 30L
        durationInput.setText(duration.toString())
        clickstreamProcessor = MobileClickstreamProcessor(model, route, duration).also { it.start() }
        liveGeneration.incrementAndGet()
        modelResultView.text = if (route == MobileProcessingRoute.FIXED_DURATION) {
            "Đang nhận clickstream · xử lý mỗi ${duration}s theo event timestamp..."
        } else {
            "Đang nhận clickstream · gom cụm ngay khi journey hoàn tất..."
        }
    }

    private fun enqueueLiveEvent(event: ClickstreamEvent) {
        val processor = clickstreamProcessor ?: return
        val generation = liveGeneration.get()
        modelExecutor.execute {
            if (generation != liveGeneration.get()) return@execute
            val update = processor.accept(event)
            if (!update.emitted) return@execute
            runOnUiThread {
                if (generation == liveGeneration.get()) {
                    renderProcessingUpdate(update)
                }
            }
        }
    }

    private fun finishLiveAnalysis() {
        val processor = clickstreamProcessor
        if (processor == null) {
            analyzeCurrentSession()
            return
        }
        val generation = liveGeneration.get()
        modelExecutor.execute {
            if (generation != liveGeneration.get()) return@execute
            val update = processor.finish()
            if (update != null) {
                runOnUiThread {
                    if (generation == liveGeneration.get()) {
                        renderProcessingUpdate(update)
                    }
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
        modelResultView.text = "Đang segment rule-based + ONNX + Markov trên ${session.events.size} events..."
        Thread {
            try {
                val result = model.analyzeEvents(session.events)
                runOnUiThread {
                    renderModelResult(result)
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

    private fun renderModelResult(result: MobileAnalysisResult) {
        updateJsonOutput(result.toJson().toString(2))
        val scored = result.predictions.filter { it.cluster != null }
        modelResultView.text = buildString {
            append("MODEL OUTPUT · ${result.predictions.size} rule-based journeys\n")
            append("scored=${scored.size} · collecting=${result.predictions.size - scored.size}\n")
            if (result.predictions.isEmpty()) {
                append("Không có journey")
            } else {
                result.predictions.takeLast(5).forEach { prediction ->
                append("${prediction.journeyId}  boundary=${prediction.boundaryReason.ifBlank { "-" }}\n")
                append("  cluster=${prediction.effectiveClusterKey ?: "-"}  ${prediction.clusterName ?: prediction.className ?: prediction.state}\n")
                append("  assignment=${prediction.assignmentType ?: "collecting"}\n")
                append("  sequence=${prediction.eventSequence.joinToString(" -> ")}\n")
                append("  flags=${prediction.frictionFlags.ifBlank { "-" }}\n")
                    append("  next=${prediction.nextAction ?: "-"}\n")
                }
            }
        }
    }

    private fun renderProcessingUpdate(update: MobileProcessingUpdate) {
        updateJsonOutput(update.toJson().toString(2))
        modelResultView.text = buildString {
            append("LIVE MODEL OUTPUT · ${update.route.wireName}\n")
            if (update.windowIndex != null) {
                append("window=${update.windowIndex}  events=${update.eventsInWindow}")
                append("  ${formatTimestamp(update.windowStartTimestamp)} -> ${formatTimestamp(update.windowEndTimestamp)}\n")
            } else if (update.windowEndTimestamp != null) {
                append("event_time=${formatTimestamp(update.windowEndTimestamp)}\n")
            }
            if (update.errors.isNotEmpty()) append("errors=${update.errors.joinToString(",")}\n")
            update.finalized.forEach { prediction ->
                append("${prediction.journeyId}  boundary=${prediction.boundaryReason.ifBlank { "flush" }}\n")
                append("  cluster=${prediction.effectiveClusterKey ?: "-"}  ${prediction.clusterName ?: prediction.className ?: prediction.state}\n")
                append("  assignment=${prediction.assignmentType ?: "collecting"}\n")
                append("  sequence=${prediction.eventSequence.joinToString(" -> ")}\n")
                append("  flags=${prediction.frictionFlags.ifBlank { "-" }}\n")
                append("  next=${prediction.nextAction ?: "-"}\n")
            }
            update.provisional.forEach { prediction ->
                append("provisional ${prediction.journeyId}  events=${prediction.eventsSeen}  state=${prediction.state}\n")
            }
            if (update.finalized.isEmpty() && update.provisional.isEmpty() && update.errors.isEmpty()) {
                append("Đã nhận window, chưa có journey đủ dài để gom cụm")
            }
        }
    }

    private fun updateJsonOutput(json: String) {
        latestJsonOutput = json
        jsonOutputView.text = json
        copyJsonButton.isEnabled = true
    }

    private fun formatTimestamp(timestamp: Long?): String = timestamp?.let {
        dateFormat.format(Date(it))
    } ?: "-"

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
