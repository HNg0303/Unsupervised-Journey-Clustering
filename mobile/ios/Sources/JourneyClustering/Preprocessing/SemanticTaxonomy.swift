import Foundation

/// Native Swift port of Python Stage 2 semantic taxonomy (taxonomy.py + semantics.py).
/// Provides multi-resolution business classification: family, module, object, operation, and funnel stage.
public enum SemanticTaxonomy {
    public static let unknown = "unknown"
    public static let general = "general"
    public static let missing = "<missing>"

    // MARK: - Normalisation Constants & Tables

    private static let dynamicRegexes: [NSRegularExpression] = [
        try! NSRegularExpression(pattern: "^\\{[a-z]+\\}$", options: []),
        try! NSRegularExpression(pattern: "^-?\\d+$", options: []),
        try! NSRegularExpression(pattern: "^[0-9a-f]{8,}$", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "^v\\d+$", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "^(null|none|nil|undefined|nan)$", options: [.caseInsensitive])
    ]

    private static let unitRegex = try! NSRegularExpression(pattern: "^\\d+[A-Za-z]+$", options: [])
    private static let wordRegex = try! NSRegularExpression(pattern: "[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z][a-z0-9]*|[0-9]+", options: [])
    private static let vcSuffixRegex = try! NSRegularExpression(pattern: "(?<=[A-Za-z])VC$", options: [])

    private static let wordAliases: [String: [String]] = [
        "fprotect": ["fsafe"],
        "protect": ["fsafe"],
        "fs": ["fsafe"],
        "pc": ["fsafe"],
        "noti": ["notification"],
        "notifications": ["notification"],
        "infor": ["info"],
        "information": ["info"],
        "details": ["detail"],
        "devices": ["device"],
        "profiles": ["profile"],
        "contracts": ["contract"],
        "vouchers": ["voucher"],
        "cards": ["card"],
        "modems": ["modem"],
        "locking": ["lock"],
        "blocking": ["block"],
        "blocked": ["block"],
        "searching": ["search"],
        "payement": ["payment"],
        "loylaty": ["loyalty"],
        "succesful": ["success"],
        "chossen": ["chosen"],
        "ap": ["access", "point"],
        "dkol": ["sale", "online"],
        "csat": ["survey"],
        "uf": ["ultra", "fast"],
        "cccd": ["vneid"],
        "napas": ["card"],
        "vietqr": ["qr"],
        "adsview": ["ads", "view"],
        "dr": ["doctor"],
        "loy": ["loyalty"]
    ]

    private static let phraseAliases: [[String]: [String]] = [
        ["wi", "fi"]: ["wifi"],
        ["e", "contract"]: ["econtract"],
        ["e", "counter"]: ["ecounter"],
        ["e", "bill"]: ["bill"],
        ["access", "point"]: ["modem"],
        ["go", "to"]: ["goto"],
        ["pull", "to", "refresh"]: ["refresh"],
        ["turn", "on", "off"]: ["toggle"],
        ["do", "action"]: ["doaction"],
        ["un", "lock"]: ["unlock"],
        ["pre", "paid"]: ["prepaid"],
        ["post", "paid"]: ["postpaid"]
    ]

    private static let infrastructureWords: Set<String> = [
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
    ]

    // MARK: - Taxonomies

    private static let strongSections: [[String]: (family: String, module: String)] = buildSectionIndex(
        taxonomy: [
            "internet": [
                general: ["internet", "service", "internet_service", "service_management", "service_manage", "manage_service", "network"],
                "modem": ["modem", "router", "wifi_router", "network_model", "rename_box", "manage_box"],
                "wifi": ["wifi", "ssid", "coverage", "band"],
                "device": ["device", "connected_device", "management_device", "device_control", "device_manager", "device_detection", "history_access", "poor_connect"],
                "parental_control": ["fsafe", "f_safe", "parental", "website", "threat", "internet_break", "harmful_content", "block_content", "content_filter", "safe_search", "trusted_device"],
                "diagnostics": ["speedtest", "network_chart", "visualize_network", "doctor_smart", "health_net", "check_game", "evaluate_coverage"],
                "schedule": ["block_schedule", "modem_schedule", "wifi_schedule", "schedule_wifi", "time_setting", "access_block", "restart_schedule"]
            ],
            "payment": [
                general: ["payment", "pay", "bill", "billing", "invoice", "cash_in", "pay_later", "wallet"],
                "prepaid": ["prepaid", "postpaid"],
                "autopay": ["autopay", "auto_pay", "payment_schedule", "schedule_payment", "list_auto_pay"],
                "method": ["payment_method", "my_card", "card", "qr_payment", "payment_qr", "paylocator"],
                "history": ["payment_history", "history_payment", "transaction_history", "payment_extension"],
                "checkout": ["checkout", "confirm_payment", "payment_info", "info_payment", "offer_payment", "payment_result", "re_payment"],
                "behalf": ["behalf", "on_behalf"],
                "gold": ["fgold", "f_gold"]
            ],
            "account": [
                general: ["account", "personal", "profile", "manage_account", "auth_account"],
                "contract": ["contract", "choose_contract", "contract_management", "manage_contract", "no_contract"],
                "econtract": ["econtract", "quote_minutes", "acceptance", "signed_success"],
                "info_change": ["ecounter", "change_info", "info_change", "change_address", "change_phone", "change_charge_address", "change_invoice_address", "kyc", "ocr", "read_card", "vneid", "scan_document"],
                "settings": ["account_setting", "setting_account", "policy", "manage_info"],
                "permission": ["decentralize", "authorization", "assign_user", "verify_employee"]
            ],
            "support": [
                general: ["support", "help"],
                "request": ["support_create", "create_request", "request_status", "support_request", "list_support", "select_type_support", "select_content_support"],
                "chat": ["chatbot", "chat", "support_chat"],
                "report": ["report", "feedback", "question_report", "filter_report", "choose_time_report"],
                "survey": ["survey", "rating", "form_survey"]
            ],
            "notification": [
                general: ["notification"],
                "settings": ["setting_notification", "notification_setting", "notification_channel", "popup_remind", "remind_notification"]
            ],
            "loyalty": [
                general: ["loyalty", "member", "point"],
                "voucher": ["voucher", "gift", "giftcode", "redeem"],
                "game": ["game", "checkin", "tarot", "shaking", "shake", "fortune", "ultra_fast"],
                "promotion": ["promotion", "promo"],
                "referral": ["refer", "referral", "refer_friends", "my_qr"]
            ],
            "shop": [
                general: ["shop", "ecommerce", "store"],
                "catalog": ["product", "product_management", "product_detail", "category"],
                "order": ["order", "order_history", "order_info", "order_detail", "cart", "address_book", "vat_info"],
                "sale": ["sale", "sale_online", "register_info", "select_contract"]
            ],
            "auth": [
                general: ["login", "logout", "sign_in", "sign_out", "guest", "otp", "explore"]
            ],
            "camera": [
                general: ["camera"]
            ],
            "tv": [
                general: ["tv"]
            ],
            "marketing": [
                "ads": ["ads", "banner", "mascot"],
                "message": ["short_msg", "full_msg", "msg", "message"]
            ]
        ]
    )

    private static let weakSections: [[String]: (family: String, module: String)] = buildSectionIndex(
        taxonomy: [
            "home": [
                general: ["home"],
                "navigation": ["nav"],
                "favourite": ["fav", "favorite", "favourite", "favorite_feature", "fav_function"],
                "customize": ["customize", "customize_function", "change_logo"]
            ]
        ]
    )

    private static let fallbackSections: [[String]: (family: String, module: String)] = buildSectionIndex(
        taxonomy: [
            "chrome": [
                "container": ["tab_bar", "alert", "scene", "window", "password_saving"],
                "popup": ["popup"],
                "boot": ["splash", "launch", "welcome"]
            ]
        ]
    )

    private static let objectPhrases: [[String]: String] = buildFlatIndex(
        taxonomy: [
            "device": ["device", "connected_device", "management_device"],
            "modem": ["modem", "router", "box", "network_model"],
            "wifi": ["wifi", "ssid", "band"],
            "profile": ["profile"],
            "contract": ["contract"],
            "econtract": ["econtract", "quote_minutes"],
            "bill": ["bill", "billing", "invoice"],
            "payment": ["payment", "checkout"],
            "card": ["card"],
            "voucher": ["voucher", "gift", "giftcode"],
            "game": ["game", "tarot", "checkin"],
            "promotion": ["promotion", "promo"],
            "notification": ["notification"],
            "schedule": ["schedule", "time_setting"],
            "request": ["request", "ticket"],
            "report": ["report", "feedback"],
            "survey": ["survey", "rating"],
            "website": ["website", "threat", "harmful_content", "content"],
            "product": ["product"],
            "order": ["order", "cart"],
            "account": ["account", "personal"],
            "otp": ["otp"],
            "password": ["password", "passwd"],
            "name": ["name"],
            "address": ["address"],
            "phone": ["phone"],
            "email": ["email"],
            "camera": ["camera"],
            "tv": ["tv"],
            "package": ["package", "prepaid", "postpaid"],
            "network": ["network"],
            "message": ["msg", "message", "chat"],
            "popup": ["popup", "banner", "ads", "mascot"],
            "qr": ["qr"],
            "history": ["history"]
        ]
    )

    private static let operationPhrases: [[String]: String] = buildFlatIndex(
        taxonomy: [
            "open": ["click", "open", "enter", "goto", "nav", "show", "doaction", "tap", "popup"],
            "view": ["view", "detail", "info", "read"],
            "list": ["list", "all", "tab_all"],
            "select": ["select", "choose", "chosen", "pick", "assign", "check"],
            "search": ["search"],
            "filter": ["filter", "tab_connected", "tab_block", "sort"],
            "refresh": ["refresh", "reload", "sync"],
            "create": ["create", "add", "new", "register"],
            "update": ["update", "edit", "change", "set", "setting", "config", "configure"],
            "rename": ["rename", "change_name", "set_name"],
            "delete": ["delete", "remove"],
            "block": ["block", "lock", "restrict"],
            "unblock": ["unblock", "unlock", "allow", "trust"],
            "enable": ["turn_on", "activate", "enable"],
            "disable": ["turn_off", "deactivate", "disable"],
            "toggle": ["toggle", "switch"],
            "restart": ["reboot", "restart", "reset"],
            "schedule": ["schedule"],
            "confirm": ["confirm", "accept", "agree", "verify", "ok"],
            "submit": ["submit", "send", "save", "apply", "upload"],
            "pay": ["pay", "checkout", "purchase", "buy"],
            "cancel": ["cancel"],
            "back": ["back", "backbutton", "handleback", "goback"],
            "close": ["close", "dismiss", "skip", "exit"],
            "login": ["login", "sign_in"],
            "logout": ["logout", "log_out", "sign_out"],
            "share": ["share"],
            "scan": ["scan", "speedtest", "health_net"],
            "download": ["download", "export"],
            "rate": ["rate", "rating", "vote"]
        ]
    )

    private static let operationStages: [String: String] = [
        "open": "entry",
        "view": "entry",
        "list": "browse",
        "select": "browse",
        "search": "browse",
        "filter": "browse",
        "refresh": "browse",
        "create": "configure",
        "update": "configure",
        "rename": "configure",
        "schedule": "configure",
        "enable": "configure",
        "disable": "configure",
        "toggle": "configure",
        "confirm": "commit",
        "submit": "commit",
        "pay": "commit",
        "delete": "commit",
        "block": "commit",
        "unblock": "commit",
        "restart": "commit",
        "login": "commit",
        "logout": "commit",
        "share": "commit",
        "scan": "commit",
        "download": "commit",
        "rate": "commit",
        "back": "abort",
        "close": "abort",
        "cancel": "abort"
    ]

    private static let genericActionPhrases: Set<[String]> = [
        ["back"], ["backbutton"], ["handleback"], ["goback"],
        ["popup"], ["confirm"], ["cancel"], ["close"],
        ["dismiss"], ["skip"], ["ok"], ["submit"],
        ["send"], ["save"], ["next"], ["continue"],
        ["refresh"], ["reload"], ["select"], ["view"]
    ]

    private static let confidenceWeights: [String: Double] = [
        "family_strong": 0.40,
        "family_weak": 0.20,
        "family_inherited": 0.15,
        "module_specific": 0.20,
        "module_general": 0.05,
        "module_inherited": 0.08,
        "object_matched": 0.15,
        "object_inherited": 0.07,
        "operation_matched": 0.25,
        "operation_default": 0.10
    ]

    // MARK: - Normalization & Path Matching

    public static func normalizePath(_ raw: String?) -> [String] {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return []
        }
        var text = raw
        let low = text.lowercased()
        if ["nan", "none", "null", "<na>", "<missing>", "<none>"].contains(low) {
            return []
        }
        if text.hasPrefix("<") && text.hasSuffix(">") {
            return []
        }

        if let qIdx = text.firstIndex(of: "?") {
            text = String(text[..<qIdx])
        }
        if let hIdx = text.firstIndex(of: "#") {
            text = String(text[..<hIdx])
        }

        var words: [String] = []
        let segments = text.components(separatedBy: CharacterSet(charactersIn: "/\\"))
        for seg in segments {
            if seg.isEmpty || isDynamic(seg) { continue }
            let parts = seg.components(separatedBy: CharacterSet.alphanumerics.inverted)
            for part in parts {
                if part.isEmpty || isDynamic(part) { continue }
                let splitWords = splitPart(part)
                for w in splitWords {
                    if isDynamic(w) { continue }
                    let expanded = wordAliases[w] ?? [w]
                    words.append(contentsOf: expanded)
                }
            }
        }

        let aliased = applyPhraseAliases(words)
        return aliased.filter { !infrastructureWords.contains($0) }
    }

    private static func isDynamic(_ text: String) -> Bool {
        let range = NSRange(location: 0, length: text.utf16.count)
        return dynamicRegexes.contains { $0.firstMatch(in: text, options: [], range: range) != nil }
    }

    private static func splitPart(_ part: String) -> [String] {
        let range = NSRange(location: 0, length: part.utf16.count)
        if unitRegex.firstMatch(in: part, options: [], range: range) != nil {
            return [part.lowercased()]
        }
        let clean = vcSuffixRegex.stringByReplacingMatches(in: part, options: [], range: range, withTemplate: "")
        let cleanRange = NSRange(location: 0, length: clean.utf16.count)
        let matches = wordRegex.matches(in: clean, options: [], range: cleanRange)
        return matches.compactMap {
            Range($0.range, in: clean).map { String(clean[$0]).lowercased() }
        }
    }

    private static func applyPhraseAliases(_ words: [String]) -> [String] {
        var out: [String] = []
        var i = 0
        let maxLen = 3
        while i < words.count {
            var matched = false
            let upper = min(maxLen, words.count - i)
            if upper >= 1 {
                for len in stride(from: upper, through: 1, by: -1) {
                    let slice = Array(words[i..<(i + len)])
                    if let replacement = phraseAliases[slice] {
                        out.append(contentsOf: replacement)
                        i += len
                        matched = true
                        break
                    }
                }
            }
            if !matched {
                out.append(words[i])
                i += 1
            }
        }
        return out
    }

    private struct Match {
        let end: Int
        let length: Int
        let phrase: [String]
    }

    private static func findMatches<T>(words: [String], index: [[String]: T], maxLength: Int) -> [(Match, T)] {
        var found: [(Match, T)] = []
        for start in 0..<words.count {
            let upper = min(maxLength, words.count - start)
            guard upper >= 1 else { continue }
            for len in 1...upper {
                let phrase = Array(words[start..<(start + len)])
                if let value = index[phrase] {
                    found.append((Match(end: start + len - 1, length: len, phrase: phrase), value))
                }
            }
        }
        return found
    }

    private static func deepest<T>(_ candidates: [(Match, T)]) -> (Match, T)? {
        guard !candidates.isEmpty else { return nil }
        return candidates.max { a, b in
            if a.0.end != b.0.end {
                return a.0.end < b.0.end
            }
            return a.0.length < b.0.length
        }
    }

    public struct SectionResult: Sendable {
        public let family: String
        public let module: String
        public let phrase: String
        public let strength: String
    }

    public static func resolveSection(_ words: [String], allowContainer: Bool = false) -> SectionResult? {
        let tiers: [(strength: String, index: [[String]: (family: String, module: String)])] = [
            ("strong", strongSections),
            ("weak", weakSections),
            ("weak", fallbackSections)
        ]

        for tier in tiers {
            let matches = findMatches(words: words, index: tier.index, maxLength: 4)
            if !matches.isEmpty {
                let specific = matches.filter { $0.1.module != general }
                let best = deepest(specific) ?? deepest(matches)
                if let best = best {
                    return SectionResult(
                        family: best.1.family,
                        module: best.1.module,
                        phrase: best.0.phrase.joined(separator: "_"),
                        strength: tier.strength
                    )
                }
            }
        }

        if allowContainer && words.allSatisfy({ operationPhrases[[$0]] != nil }) {
            return SectionResult(family: "chrome", module: "container", phrase: "container_only", strength: "weak")
        }
        return nil
    }

    public static func resolveObject(_ words: [String]) -> (obj: String, phrase: String)? {
        let best = deepest(findMatches(words: words, index: objectPhrases, maxLength: 3))
        return best.map { ($0.1, $0.0.phrase.joined(separator: "_")) }
    }

    public static func resolveOperation(_ words: [String]) -> (operation: String, phrase: String)? {
        let best = deepest(findMatches(words: words, index: operationPhrases, maxLength: 3))
        return best.map { ($0.1, $0.0.phrase.joined(separator: "_")) }
    }

    // MARK: - Classification API

    public final class LabelBuilder {
        public var family: String = unknown
        public var module: String = unknown
        public var obj: String = unknown
        public var operation: String = unknown
        public var confidence: Double = 0.0
        public var evidence: [String] = []

        public init() {}

        public func credit(weightKey: String, slot: String, value: String, source: String, phrase: String) {
            confidence += confidenceWeights[weightKey] ?? 0.0
            evidence.append("\(slot)=\(value)<-\(source):\(phrase)")
        }

        public func build() -> SemanticLabel {
            guard family != unknown else { return SemanticLabel() }
            let stage = operationStages[operation] ?? unknown
            let roundedConf = round(min(1.0, confidence) * 1000.0) / 1000.0
            return SemanticLabel(
                businessFamily: family,
                businessModule: module,
                businessObject: obj,
                operation: operation,
                operationStage: stage,
                semanticConfidence: roundedConf,
                semanticEvidence: evidence.joined(separator: ";")
            )
        }
    }

    public static func classifyScreen(_ screen: String?) -> SemanticLabel {
        guard let screen = screen, !screen.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return SemanticLabel()
        }
        let words = normalizePath(screen)
        let builder = LabelBuilder()
        guard let section = resolveSection(words, allowContainer: true) else {
            return SemanticLabel()
        }

        builder.family = section.family
        builder.module = section.module
        builder.credit(weightKey: "family_\(section.strength)", slot: "family", value: section.family, source: "screen", phrase: section.phrase)
        builder.credit(weightKey: section.module == general ? "module_general" : "module_specific", slot: "module", value: section.module, source: "screen", phrase: section.phrase)

        if let matchedObj = resolveObject(words) {
            builder.obj = matchedObj.obj
            builder.credit(weightKey: "object_matched", slot: "object", value: matchedObj.obj, source: "screen", phrase: matchedObj.phrase)
        }

        if let matchedOp = resolveOperation(words) {
            builder.operation = matchedOp.operation
            builder.credit(weightKey: "operation_matched", slot: "operation", value: matchedOp.operation, source: "screen", phrase: matchedOp.phrase)
        } else {
            builder.operation = "view"
            builder.credit(weightKey: "operation_default", slot: "operation", value: "view", source: "screen", phrase: "event_type")
        }
        return builder.build()
    }

    public static func classifyAction(target: String?, context: SemanticLabel) -> SemanticLabel {
        let words = normalizePath(target)
        let builder = LabelBuilder()
        let isGeneric = genericActionPhrases.contains(words)
        let section = isGeneric ? nil : resolveSection(words, allowContainer: false)

        if let section = section {
            builder.family = section.family
            builder.module = section.module
            builder.credit(weightKey: "family_\(section.strength)", slot: "family", value: section.family, source: "target", phrase: section.phrase)
            builder.credit(weightKey: section.module == general ? "module_general" : "module_specific", slot: "module", value: section.module, source: "target", phrase: section.phrase)
        } else if context.isKnown {
            builder.family = context.businessFamily
            builder.module = context.businessModule
            builder.credit(weightKey: "family_inherited", slot: "family", value: context.businessFamily, source: "context", phrase: "screen_context")
            builder.credit(weightKey: "module_inherited", slot: "module", value: context.businessModule, source: "context", phrase: "screen_context")
        }

        let matchedObj = isGeneric ? nil : resolveObject(words)
        if let matchedObj = matchedObj {
            builder.obj = matchedObj.obj
            builder.credit(weightKey: "object_matched", slot: "object", value: matchedObj.obj, source: "target", phrase: matchedObj.phrase)
        } else if context.businessObject != unknown {
            builder.obj = context.businessObject
            builder.credit(weightKey: "object_inherited", slot: "object", value: context.businessObject, source: "context", phrase: "screen_context")
        }

        if let matchedOp = resolveOperation(words) {
            builder.operation = matchedOp.operation
            builder.credit(weightKey: "operation_matched", slot: "operation", value: matchedOp.operation, source: "target", phrase: matchedOp.phrase)
        }
        return builder.build()
    }

    public static func classifyEvent(eventType: String, screenContext: String, segmentName: String) -> SemanticLabel {
        let context = classifyScreen(screenContext)
        if eventType.lowercased() == "view" {
            return context
        } else {
            return classifyAction(target: segmentName, context: context)
        }
    }

    // MARK: - Private Index Builders

    private static func buildFlatIndex(taxonomy: [String: [String]]) -> [[String]: String] {
        var index: [[String]: String] = [:]
        for (label, phrases) in taxonomy {
            for phrase in phrases {
                let words = phrase.components(separatedBy: "_").filter { !$0.isEmpty }
                index[words] = label
            }
        }
        return index
    }

    private static func buildSectionIndex(taxonomy: [String: [String: [String]]]) -> [[String]: (family: String, module: String)] {
        var index: [[String]: (family: String, module: String)] = [:]
        for (family, modules) in taxonomy {
            for (module, phrases) in modules {
                for phrase in phrases {
                    let words = phrase.components(separatedBy: "_").filter { !$0.isEmpty }
                    index[words] = (family, module)
                }
            }
        }
        return index
    }
}
