package vn.hifpt.clickstream

import java.net.URI
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.regex.Pattern
import kotlin.math.min

/**
 * Native port of Python Stage 2 semantic taxonomy (taxonomy.py + semantics.py).
 * Provides multi-resolution business classification: family, module, object, operation, and funnel stage.
 */
object MobileSemanticTaxonomy {
    const val UNKNOWN = "unknown"
    const val GENERAL = "general"
    const val MISSING = "<missing>"

    // --------------------------------------------------------------------------
    // Normalisation regexes and tables
    // --------------------------------------------------------------------------
    private val dynamicPatterns = listOf(
        Regex("^\\{[a-z]+\\}$"),
        Regex("^-?\\d+$"),
        Regex("^[0-9a-f]{8,}$", RegexOption.IGNORE_CASE),
        Regex("^v\\d+$", RegexOption.IGNORE_CASE),
        Regex("^(null|none|nil|undefined|nan)$", RegexOption.IGNORE_CASE)
    )

    private val segmentSplitPattern = Pattern.compile("[/\\\\]+")
    private val partSplitPattern = Pattern.compile("[^0-9A-Za-z]+")
    private val unitPattern = Pattern.compile("^\\d+[A-Za-z]+$")
    private val wordPattern = Pattern.compile("[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z][a-z0-9]*|[0-9]+")
    private val vcSuffixPattern = Pattern.compile("(?<=[A-Za-z])VC$")

    private val semanticQueryKeys = setOf(
        "tab", "cat_id", "categoryid", "ordertype", "type", "step", "mode", "view", "status"
    )

    private val wordAliases: Map<String, List<String>> = mapOf(
        "fprotect" to listOf("fsafe"),
        "protect" to listOf("fsafe"),
        "fs" to listOf("fsafe"),
        "pc" to listOf("fsafe"),
        "noti" to listOf("notification"),
        "notifications" to listOf("notification"),
        "infor" to listOf("info"),
        "information" to listOf("info"),
        "details" to listOf("detail"),
        "devices" to listOf("device"),
        "profiles" to listOf("profile"),
        "contracts" to listOf("contract"),
        "vouchers" to listOf("voucher"),
        "cards" to listOf("card"),
        "modems" to listOf("modem"),
        "locking" to listOf("lock"),
        "blocking" to listOf("block"),
        "blocked" to listOf("block"),
        "searching" to listOf("search"),
        "payement" to listOf("payment"),
        "loylaty" to listOf("loyalty"),
        "succesful" to listOf("success"),
        "chossen" to listOf("chosen"),
        "ap" to listOf("access", "point"),
        "dkol" to listOf("sale", "online"),
        "csat" to listOf("survey"),
        "uf" to listOf("ultra", "fast"),
        "cccd" to listOf("vneid"),
        "napas" to listOf("card"),
        "vietqr" to listOf("qr"),
        "adsview" to listOf("ads", "view"),
        "dr" to listOf("doctor"),
        "loy" to listOf("loyalty")
    )

    private val phraseAliases: Map<List<String>, List<String>> = mapOf(
        listOf("wi", "fi") to listOf("wifi"),
        listOf("e", "contract") to listOf("econtract"),
        listOf("e", "counter") to listOf("ecounter"),
        listOf("e", "bill") to listOf("bill"),
        listOf("access", "point") to listOf("modem"),
        listOf("go", "to") to listOf("goto"),
        listOf("pull", "to", "refresh") to listOf("refresh"),
        listOf("turn", "on", "off") to listOf("toggle"),
        listOf("do", "action") to listOf("doaction"),
        listOf("un", "lock") to listOf("unlock"),
        listOf("pre", "paid") to listOf("prepaid"),
        listOf("post", "paid") to listOf("postpaid")
    )

    private val infrastructureWords = setOf(
        "android", "ios", "app", "hi", "fpt", "vn", "com", "www", "uri",
        "http", "https", "url", "web", "webview", "webpage", "webkit", "api", "sm",
        "vc", "controller", "activity", "fragment", "screen", "page", "host",
        "hosting", "ui", "uikit", "sdk", "base", "main", "st", "sf", "platform",
        "introspection", "anchor", "remote", "browser", "safari", "authentication",
        "navigation", "tabbar", "keyboard", "overlay", "toast", "loading",
        "dexter", "redirect", "receiver", "tracking", "element", "system",
        "alternate", "application", "icons", "with", "inui", "voice",
        "shortcut", "hidden", "input", "basic", "pdf", "non", "interaction",
        "header", "footer", "body", "item", "icon", "bottom", "sheet", "button",
        "custom", "size", "big", "sa", "os", "data", "default", "action",
        "btn", "v2", "hifpt", "ftel"
    )

    // --------------------------------------------------------------------------
    // Taxonomies
    // --------------------------------------------------------------------------
    private val strongSections = buildSectionIndex(
        mapOf(
            "internet" to mapOf(
                GENERAL to listOf("internet", "service", "internet_service", "service_management", "service_manage", "manage_service", "network"),
                "modem" to listOf("modem", "router", "wifi_router", "network_model", "rename_box", "manage_box"),
                "wifi" to listOf("wifi", "ssid", "coverage", "band"),
                "device" to listOf("device", "connected_device", "management_device", "device_control", "device_manager", "device_detection", "history_access", "poor_connect"),
                "parental_control" to listOf("fsafe", "f_safe", "parental", "website", "threat", "internet_break", "harmful_content", "block_content", "content_filter", "safe_search", "trusted_device"),
                "diagnostics" to listOf("speedtest", "network_chart", "visualize_network", "doctor_smart", "health_net", "check_game", "evaluate_coverage"),
                "schedule" to listOf("block_schedule", "modem_schedule", "wifi_schedule", "schedule_wifi", "time_setting", "access_block", "restart_schedule")
            ),
            "payment" to mapOf(
                GENERAL to listOf("payment", "pay", "bill", "billing", "invoice", "cash_in", "pay_later", "wallet"),
                "prepaid" to listOf("prepaid", "postpaid"),
                "autopay" to listOf("autopay", "auto_pay", "payment_schedule", "schedule_payment", "list_auto_pay"),
                "method" to listOf("payment_method", "my_card", "card", "qr_payment", "payment_qr", "paylocator"),
                "history" to listOf("payment_history", "history_payment", "transaction_history", "payment_extension"),
                "checkout" to listOf("checkout", "confirm_payment", "payment_info", "info_payment", "offer_payment", "payment_result", "re_payment"),
                "behalf" to listOf("behalf", "on_behalf"),
                "gold" to listOf("fgold", "f_gold")
            ),
            "account" to mapOf(
                GENERAL to listOf("account", "personal", "profile", "manage_account", "auth_account"),
                "contract" to listOf("contract", "choose_contract", "contract_management", "manage_contract", "no_contract"),
                "econtract" to listOf("econtract", "quote_minutes", "acceptance", "signed_success"),
                "info_change" to listOf("ecounter", "change_info", "info_change", "change_address", "change_phone", "change_charge_address", "change_invoice_address", "kyc", "ocr", "read_card", "vneid", "scan_document"),
                "settings" to listOf("account_setting", "setting_account", "policy", "manage_info"),
                "permission" to listOf("decentralize", "authorization", "assign_user", "verify_employee")
            ),
            "support" to mapOf(
                GENERAL to listOf("support", "help"),
                "request" to listOf("support_create", "create_request", "request_status", "support_request", "list_support", "select_type_support", "select_content_support"),
                "chat" to listOf("chatbot", "chat", "support_chat"),
                "report" to listOf("report", "feedback", "question_report", "filter_report", "choose_time_report"),
                "survey" to listOf("survey", "rating", "form_survey")
            ),
            "notification" to mapOf(
                GENERAL to listOf("notification"),
                "settings" to listOf("setting_notification", "notification_setting", "notification_channel", "popup_remind", "remind_notification")
            ),
            "loyalty" to mapOf(
                GENERAL to listOf("loyalty", "member", "point"),
                "voucher" to listOf("voucher", "gift", "giftcode", "redeem"),
                "game" to listOf("game", "checkin", "tarot", "shaking", "shake", "fortune", "ultra_fast"),
                "promotion" to listOf("promotion", "promo"),
                "referral" to listOf("refer", "referral", "refer_friends", "my_qr")
            ),
            "shop" to mapOf(
                GENERAL to listOf("shop", "ecommerce", "store"),
                "catalog" to listOf("product", "product_management", "product_detail", "category"),
                "order" to listOf("order", "order_history", "order_info", "order_detail", "cart", "address_book", "vat_info"),
                "sale" to listOf("sale", "sale_online", "register_info", "select_contract")
            ),
            "auth" to mapOf(
                GENERAL to listOf("login", "logout", "sign_in", "sign_out", "guest", "otp", "explore")
            ),
            "camera" to mapOf(
                GENERAL to listOf("camera")
            ),
            "tv" to mapOf(
                GENERAL to listOf("tv")
            ),
            "marketing" to mapOf(
                "ads" to listOf("ads", "banner", "mascot"),
                "message" to listOf("short_msg", "full_msg", "msg", "message")
            )
        )
    )

    private val weakSections = buildSectionIndex(
        mapOf(
            "home" to mapOf(
                GENERAL to listOf("home"),
                "navigation" to listOf("nav"),
                "favourite" to listOf("fav", "favorite", "favourite", "favorite_feature", "fav_function"),
                "customize" to listOf("customize", "customize_function", "change_logo")
            )
        )
    )

    private val fallbackSections = buildSectionIndex(
        mapOf(
            "chrome" to mapOf(
                "container" to listOf("tab_bar", "alert", "scene", "window", "password_saving"),
                "popup" to listOf("popup"),
                "boot" to listOf("splash", "launch", "welcome")
            )
        )
    )

    private val objectPhrases = buildFlatIndex(
        mapOf(
            "device" to listOf("device", "connected_device", "management_device"),
            "modem" to listOf("modem", "router", "box", "network_model"),
            "wifi" to listOf("wifi", "ssid", "band"),
            "profile" to listOf("profile"),
            "contract" to listOf("contract"),
            "econtract" to listOf("econtract", "quote_minutes"),
            "bill" to listOf("bill", "billing", "invoice"),
            "payment" to listOf("payment", "checkout"),
            "card" to listOf("card"),
            "voucher" to listOf("voucher", "gift", "giftcode"),
            "game" to listOf("game", "tarot", "checkin"),
            "promotion" to listOf("promotion", "promo"),
            "notification" to listOf("notification"),
            "schedule" to listOf("schedule", "time_setting"),
            "request" to listOf("request", "ticket"),
            "report" to listOf("report", "feedback"),
            "survey" to listOf("survey", "rating"),
            "website" to listOf("website", "threat", "harmful_content", "content"),
            "product" to listOf("product"),
            "order" to listOf("order", "cart"),
            "account" to listOf("account", "personal"),
            "otp" to listOf("otp"),
            "password" to listOf("password", "passwd"),
            "name" to listOf("name"),
            "address" to listOf("address"),
            "phone" to listOf("phone"),
            "email" to listOf("email"),
            "camera" to listOf("camera"),
            "tv" to listOf("tv"),
            "package" to listOf("package", "prepaid", "postpaid"),
            "network" to listOf("network"),
            "message" to listOf("msg", "message", "chat"),
            "popup" to listOf("popup", "banner", "ads", "mascot"),
            "qr" to listOf("qr"),
            "history" to listOf("history")
        )
    )

    private val operationPhrases = buildFlatIndex(
        mapOf(
            "open" to listOf("click", "open", "enter", "goto", "nav", "show", "doaction", "tap", "popup"),
            "view" to listOf("view", "detail", "info", "read"),
            "list" to listOf("list", "all", "tab_all"),
            "select" to listOf("select", "choose", "chosen", "pick", "assign", "check"),
            "search" to listOf("search"),
            "filter" to listOf("filter", "tab_connected", "tab_block", "sort"),
            "refresh" to listOf("refresh", "reload", "sync"),
            "create" to listOf("create", "add", "new", "register"),
            "update" to listOf("update", "edit", "change", "set", "setting", "config", "configure"),
            "rename" to listOf("rename", "change_name", "set_name"),
            "delete" to listOf("delete", "remove"),
            "block" to listOf("block", "lock", "restrict"),
            "unblock" to listOf("unblock", "unlock", "allow", "trust"),
            "enable" to listOf("turn_on", "activate", "enable"),
            "disable" to listOf("turn_off", "deactivate", "disable"),
            "toggle" to listOf("toggle", "switch"),
            "restart" to listOf("reboot", "restart", "reset"),
            "schedule" to listOf("schedule"),
            "confirm" to listOf("confirm", "accept", "agree", "verify", "ok"),
            "submit" to listOf("submit", "send", "save", "apply", "upload"),
            "pay" to listOf("pay", "checkout", "purchase", "buy"),
            "cancel" to listOf("cancel"),
            "back" to listOf("back", "backbutton", "handleback", "goback"),
            "close" to listOf("close", "dismiss", "skip", "exit"),
            "login" to listOf("login", "sign_in"),
            "logout" to listOf("logout", "log_out", "sign_out"),
            "share" to listOf("share"),
            "scan" to listOf("scan", "speedtest", "health_net"),
            "download" to listOf("download", "export"),
            "rate" to listOf("rate", "rating", "vote")
        )
    )

    private val operationStages: Map<String, String> = mapOf(
        "open" to "entry",
        "view" to "entry",
        "list" to "browse",
        "select" to "browse",
        "search" to "browse",
        "filter" to "browse",
        "refresh" to "browse",
        "create" to "configure",
        "update" to "configure",
        "rename" to "configure",
        "schedule" to "configure",
        "enable" to "configure",
        "disable" to "configure",
        "toggle" to "configure",
        "confirm" to "commit",
        "submit" to "commit",
        "pay" to "commit",
        "delete" to "commit",
        "block" to "commit",
        "unblock" to "commit",
        "restart" to "commit",
        "login" to "commit",
        "logout" to "commit",
        "share" to "commit",
        "scan" to "commit",
        "download" to "commit",
        "rate" to "commit",
        "back" to "abort",
        "close" to "abort",
        "cancel" to "abort"
    )

    private val genericActionPhrases: Set<List<String>> = setOf(
        listOf("back"), listOf("backbutton"), listOf("handleback"), listOf("goback"),
        listOf("popup"), listOf("confirm"), listOf("cancel"), listOf("close"),
        listOf("dismiss"), listOf("skip"), listOf("ok"), listOf("submit"),
        listOf("send"), listOf("save"), listOf("next"), listOf("continue"),
        listOf("refresh"), listOf("reload"), listOf("select"), listOf("view")
    )

    private val confidenceWeights = mapOf(
        "family_strong" to 0.40,
        "family_weak" to 0.20,
        "family_inherited" to 0.15,
        "module_specific" to 0.20,
        "module_general" to 0.05,
        "module_inherited" to 0.08,
        "object_matched" to 0.15,
        "object_inherited" to 0.07,
        "operation_matched" to 0.25,
        "operation_default" to 0.10
    )

    // --------------------------------------------------------------------------
    // Normalization & Path Matching
    // --------------------------------------------------------------------------

    fun normalizePath(raw: String?): List<String> {
        if (raw.isNullOrBlank()) return emptyList()
        var text = raw.trim()
        val low = text.lowercase(Locale.US)
        if (low in setOf("nan", "none", "null", "<na>", "<missing>", "<none>")) return emptyList()
        if (text.startsWith("<") && text.endsWith(">")) return emptyList()

        // Strip query string and fragments for path matching
        val qIdx = text.indexOf('?')
        if (qIdx >= 0) text = text.substring(0, qIdx)
        val hIdx = text.indexOf('#')
        if (hIdx >= 0) text = text.substring(0, hIdx)

        val words = mutableListOf<String>()
        val segments = segmentSplitPattern.split(text)
        for (seg in segments) {
            if (seg.isEmpty() || isDynamic(seg)) continue
            val parts = partSplitPattern.split(seg)
            for (part in parts) {
                if (part.isEmpty() || isDynamic(part)) continue
                val splitWords = splitPart(part)
                for (w in splitWords) {
                    if (isDynamic(w)) continue
                    val expanded = wordAliases[w] ?: listOf(w)
                    words.addAll(expanded)
                }
            }
        }

        val aliased = applyPhraseAliases(words)
        return aliased.filter { it !in infrastructureWords }
    }

    private fun isDynamic(text: String): Boolean =
        dynamicPatterns.any { it.matches(text) }

    private fun splitPart(part: String): List<String> {
        if (unitPattern.matcher(part).matches()) {
            return listOf(part.lowercase(Locale.US))
        }
        val clean = vcSuffixPattern.matcher(part).replaceFirst("")
        val matcher = wordPattern.matcher(clean)
        val list = mutableListOf<String>()
        while (matcher.find()) {
            list.add(matcher.group().lowercase(Locale.US))
        }
        return list
    }

    private fun applyPhraseAliases(words: List<String>): List<String> {
        val out = mutableListOf<String>()
        var i = 0
        val maxLen = 3
        while (i < words.size) {
            var matched = false
            for (len in min(maxLen, words.size - i) downTo 1) {
                val slice = words.subList(i, i + len)
                val replacement = phraseAliases[slice]
                if (replacement != null) {
                    out.addAll(replacement)
                    i += len
                    matched = true
                    break
                }
            }
            if (!matched) {
                out.add(words[i])
                i += 1
            }
        }
        return out
    }

    private data class Match(val end: Int, val length: Int, val phrase: List<String>)

    private fun <T> findMatches(words: List<String>, index: Map<List<String>, T>, maxLength: Int): List<Pair<Match, T>> {
        val found = mutableListOf<Pair<Match, T>>()
        for (start in words.indices) {
            for (len in 1..min(maxLength, words.size - start)) {
                val phrase = words.subList(start, start + len)
                val value = index[phrase]
                if (value != null) {
                    found.add(Pair(Match(start + len - 1, len, phrase), value))
                }
            }
        }
        return found
    }

    private fun <T> deepest(candidates: List<Pair<Match, T>>): Pair<Match, T>? {
        if (candidates.isEmpty()) return null
        return candidates.maxWithOrNull(compareBy<Pair<Match, T>> { it.first.end }.thenBy { it.first.length })
    }

    fun resolveSection(words: List<String>, allowContainer: Boolean = false): SectionResult? {
        val tiers = listOf(
            "strong" to strongSections,
            "weak" to weakSections,
            "weak" to fallbackSections
        )

        for ((strength, index) in tiers) {
            val matches = findMatches(words, index, 4)
            if (matches.isNotEmpty()) {
                val specific = matches.filter { it.second.second != GENERAL }
                val best = deepest(specific) ?: deepest(matches)
                if (best != null) {
                    return SectionResult(best.second.first, best.second.second, best.first.phrase.joinToString("_"), strength)
                }
            }
        }

        if (allowContainer && words.all { operationPhrases.containsKey(listOf(it)) }) {
            return SectionResult("chrome", "container", "container_only", "weak")
        }
        return null
    }

    fun resolveObject(words: List<String>): Pair<String, String>? {
        val best = deepest(findMatches(words, objectPhrases, 3))
        return best?.let { Pair(it.second, it.first.phrase.joinToString("_")) }
    }

    fun resolveOperation(words: List<String>): Pair<String, String>? {
        val best = deepest(findMatches(words, operationPhrases, 3))
        return best?.let { Pair(it.second, it.first.phrase.joinToString("_")) }
    }

    data class SectionResult(val family: String, val module: String, val phrase: String, val strength: String)

    // --------------------------------------------------------------------------
    // Semantic Classification API
    // --------------------------------------------------------------------------

    class LabelBuilder {
        var family: String = UNKNOWN
        var module: String = UNKNOWN
        var obj: String = UNKNOWN
        var operation: String = UNKNOWN
        var confidence: Double = 0.0
        val evidence = mutableListOf<String>()

        fun credit(weightKey: String, slot: String, value: String, source: String, phrase: String) {
            confidence += confidenceWeights[weightKey] ?: 0.0
            evidence.add("$slot=$value<-$source:$phrase")
        }

        fun build(): SemanticLabel {
            if (family == UNKNOWN) return SemanticLabel()
            val stage = operationStages[operation] ?: UNKNOWN
            return SemanticLabel(
                businessFamily = family,
                businessModule = module,
                businessObject = obj,
                operation = operation,
                operationStage = stage,
                semanticConfidence = (Math.round(min(1.0, confidence) * 1000.0) / 1000.0),
                semanticEvidence = evidence.joinToString(";")
            )
        }
    }

    fun classifyScreen(screen: String?): SemanticLabel {
        if (screen.isNullOrBlank()) return SemanticLabel()
        val words = normalizePath(screen)
        val builder = LabelBuilder()
        val section = resolveSection(words, allowContainer = true) ?: return SemanticLabel()
        builder.family = section.family
        builder.module = section.module
        builder.credit("family_${section.strength}", "family", section.family, "screen", section.phrase)
        builder.credit(if (section.module == GENERAL) "module_general" else "module_specific", "module", section.module, "screen", section.phrase)

        val matchedObj = resolveObject(words)
        if (matchedObj != null) {
            builder.obj = matchedObj.first
            builder.credit("object_matched", "object", matchedObj.first, "screen", matchedObj.second)
        }

        val matchedOp = resolveOperation(words)
        if (matchedOp != null) {
            builder.operation = matchedOp.first
            builder.credit("operation_matched", "operation", matchedOp.first, "screen", matchedOp.second)
        } else {
            builder.operation = "view"
            builder.credit("operation_default", "operation", "view", "screen", "event_type")
        }
        return builder.build()
    }

    fun classifyAction(target: String?, context: SemanticLabel): SemanticLabel {
        val words = normalizePath(target)
        val builder = LabelBuilder()
        val isGeneric = genericActionPhrases.contains(words)
        val section = if (isGeneric) null else resolveSection(words, allowContainer = false)

        if (section != null) {
            builder.family = section.family
            builder.module = section.module
            builder.credit("family_${section.strength}", "family", section.family, "target", section.phrase)
            builder.credit(if (section.module == GENERAL) "module_general" else "module_specific", "module", section.module, "target", section.phrase)
        } else if (context.isKnown) {
            builder.family = context.businessFamily
            builder.module = context.businessModule
            builder.credit("family_inherited", "family", context.businessFamily, "context", "screen_context")
            builder.credit("module_inherited", "module", context.businessModule, "context", "screen_context")
        }

        val matchedObj = if (isGeneric) null else resolveObject(words)
        if (matchedObj != null) {
            builder.obj = matchedObj.first
            builder.credit("object_matched", "object", matchedObj.first, "target", matchedObj.second)
        } else if (context.businessObject != UNKNOWN) {
            builder.obj = context.businessObject
            builder.credit("object_inherited", "object", context.businessObject, "context", "screen_context")
        }

        val matchedOp = resolveOperation(words)
        if (matchedOp != null) {
            builder.operation = matchedOp.first
            builder.credit("operation_matched", "operation", matchedOp.first, "target", matchedOp.second)
        }
        return builder.build()
    }

    fun classifyEvent(eventType: String, screenContext: String, segmentName: String): SemanticLabel {
        val context = classifyScreen(screenContext)
        return if (eventType.lowercase(Locale.US) == "view") {
            context
        } else {
            classifyAction(segmentName, context)
        }
    }

    private fun buildFlatIndex(taxonomy: Map<String, List<String>>): Map<List<String>, String> {
        val index = mutableMapOf<List<String>, String>()
        for ((label, phrases) in taxonomy) {
            for (phrase in phrases) {
                val words = phrase.split("_").filter { it.isNotEmpty() }
                index[words] = label
            }
        }
        return index
    }

    private fun buildSectionIndex(taxonomy: Map<String, Map<String, List<String>>>): Map<List<String>, Pair<String, String>> {
        val index = mutableMapOf<List<String>, Pair<String, String>>()
        for ((family, modules) in taxonomy) {
            for ((module, phrases) in modules) {
                for (phrase in phrases) {
                    val words = phrase.split("_").filter { it.isNotEmpty() }
                    index[words] = Pair(family, module)
                }
            }
        }
        return index
    }
}
