import Foundation

/// Multi-Channel SVD*IDF + Numeric Scaling + Global PCA Journey Vectorizer.
public final class JourneyVectorizer: @unchecked Sendable {
    private let vocabularies: [String: [String: Int]]
    private let channelWeights: [String: Float]
    private let numericBlockWeight: Float
    private let numericColumns: [String]
    private let numericLog1pColumns: Set<String>
    private let numericMean: [Float]
    private let numericScale: [Float]
    private let primaryW: [[Float]] // [48, vocab_primary]
    private let channelW: [String: [[Float]]] // channel -> [48, vocab_ch]
    private let globalPcaComp: [[Float]]? // [48, inDim]
    private let globalPcaMean: [Float]? // [inDim]

    public let embeddingDim: Int

    public init(
        vocabularies: [String: [String: Int]],
        channelWeights: [String: Float],
        numericBlockWeight: Float,
        numericColumns: [String],
        numericLog1pColumns: Set<String>,
        numericMean: [Float],
        numericScale: [Float],
        primaryW: [[Float]],
        channelW: [String: [[Float]]],
        globalPcaComp: [[Float]]?,
        globalPcaMean: [Float]?
    ) {
        self.vocabularies = vocabularies
        self.channelWeights = channelWeights
        self.numericBlockWeight = numericBlockWeight
        self.numericColumns = numericColumns
        self.numericLog1pColumns = numericLog1pColumns
        self.numericMean = numericMean
        self.numericScale = numericScale
        self.primaryW = primaryW
        self.channelW = channelW
        self.globalPcaComp = globalPcaComp
        self.globalPcaMean = globalPcaMean
        self.embeddingDim = globalPcaComp?.count ?? 48
    }

    public func transform(journey: PreparedJourney) -> JourneyFeatureMatrix {
        let primaryEmb = encodeChannel(sequence: journey.tokens, channelName: "primary")
        var channelEmbs: [String: [Float]] = [:]
        for (chName, tokens) in journey.channels {
            if chName != "primary" && channelW[chName] != nil {
                channelEmbs[chName] = encodeChannel(sequence: tokens, channelName: chName)
            }
        }

        let numericScaled = encodeNumeric(metrics: journey.numeric)

        // Concatenate blocks: primary + sorted channels + numeric
        let sortedChannelKeys = channelW.keys.sorted()
        let totalLen = 48 * (1 + sortedChannelKeys.count) + numericColumns.count
        var concatenated = [Float](repeating: 0.0, count: totalLen)
        var offset = 0

        // 1. Primary
        let wPrim = channelWeights["primary"] ?? 1.0
        for i in 0..<48 {
            concatenated[offset + i] = primaryEmb[i] * wPrim
        }
        offset += 48

        // 2. Sorted channels
        for chName in sortedChannelKeys {
            let emb = channelEmbs[chName] ?? [Float](repeating: 0.0, count: 48)
            let wCh = channelWeights[chName] ?? 1.0
            for i in 0..<48 {
                concatenated[offset + i] = emb[i] * wCh
            }
            offset += 48
        }

        // 3. Numeric block
        for i in 0..<numericScaled.count {
            concatenated[offset + i] = numericScaled[i] * numericBlockWeight
        }

        // 4. Global PCA projection
        var projected: [Float]
        if let pcaComp = globalPcaComp, let pcaMean = globalPcaMean {
            let pcaDim = pcaComp.count
            projected = [Float](repeating: 0.0, count: pcaDim)
            for compIdx in 0..<pcaDim {
                let compRow = pcaComp[compIdx]
                var dot: Float = 0.0
                for i in 0..<concatenated.count {
                    dot += (concatenated[i] - pcaMean[i]) * compRow[i]
                }
                projected[compIdx] = dot
            }
            Float16Helper.l2Normalize(&projected)
        } else {
            projected = concatenated
            Float16Helper.l2Normalize(&projected)
        }

        return JourneyFeatureMatrix(
            primaryEmbedding: primaryEmb,
            channelEmbeddings: channelEmbs,
            scaledNumericBlock: numericScaled,
            concatenatedVector: concatenated,
            projectedEmbedding: projected
        )
    }

    private func encodeChannel(sequence: [String], channelName: String) -> [Float] {
        guard let vocab = vocabularies[channelName] else {
            return [Float](repeating: 0.0, count: 48)
        }
        let W: [[Float]]
        if channelName == "primary" {
            W = primaryW
        } else if let chMat = channelW[channelName] {
            W = chMat
        } else {
            return [Float](repeating: 0.0, count: 48)
        }

        var bounded: [String] = []
        bounded.reserveCapacity(sequence.count + 2)
        bounded.append("<bos>")
        bounded.append(contentsOf: sequence)
        bounded.append("<eos>")

        // Extract 1-grams and 2-grams
        var counts: [String: Int] = [:]
        for n in 1...2 {
            if bounded.count >= n {
                for i in 0...(bounded.count - n) {
                    let ngram = (n == 1) ? bounded[i] : "\(bounded[i]) \(bounded[i + 1])"
                    counts[ngram, default: 0] += 1
                }
            }
        }

        var z = [Float](repeating: 0.0, count: 48)
        for (ngram, cnt) in counts {
            guard let idx = vocab[ngram] else { continue }
            let tf = Float(1.0 + log(Double(cnt)))
            for comp in 0..<48 {
                z[comp] += tf * W[comp][idx]
            }
        }

        Float16Helper.l2Normalize(&z)
        return z
    }

    private func encodeNumeric(metrics: [String: Double]) -> [Float] {
        var result = [Float](repeating: 0.0, count: numericColumns.count)
        for i in 0..<numericColumns.count {
            let col = numericColumns[i]
            var raw = metrics[col] ?? 0.0
            if numericLog1pColumns.contains(col) {
                raw = log(max(0.0, raw) + 1.0)
            }
            let scale = (numericScale[i] == 0.0) ? 1.0 : numericScale[i]
            result[i] = (Float(raw) - numericMean[i]) / scale
        }
        Float16Helper.l2Normalize(&result)
        return result
    }

    // MARK: - Loading from Bundled Assets

    public static func load(
        vocabJson: String,
        metaJson: String,
        weightsData: Data
    ) throws -> JourneyVectorizer {
        guard let vocabData = vocabJson.data(using: .utf8),
              let vocabObj = try JSONSerialization.jsonObject(with: vocabData) as? [String: [String: Int]] else {
            throw NSError(domain: "JourneyVectorizer", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid vocabularies.json"])
        }

        guard let metaData = metaJson.data(using: .utf8),
              let metaObj = try JSONSerialization.jsonObject(with: metaData) as? [String: Any] else {
            throw NSError(domain: "JourneyVectorizer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid vectorizer_metadata.json"])
        }

        let cwObj = (metaObj["channel_weights"] as? [String: Double]) ?? [:]
        var channelWeights: [String: Float] = [:]
        for (k, v) in cwObj { channelWeights[k] = Float(v) }

        let numericBlockWeight = Float(metaObj["numeric_block_weight"] as? Double ?? 0.35)
        let numericColumns = (metaObj["numeric_columns"] as? [String]) ?? []
        let numericLog1pColumns = Set((metaObj["numeric_log1p_columns"] as? [String]) ?? [])
        let numericMean = ((metaObj["numeric_mean"] as? [Double]) ?? []).map { Float($0) }
        let numericScale = ((metaObj["numeric_scale"] as? [Double]) ?? []).map { Float($0) }

        let hasGlobalPca = (metaObj["has_global_pca"] as? Bool) ?? false
        let pcaDim = hasGlobalPca ? ((metaObj["global_pca_components_count"] as? Int) ?? 48) : 0

        var cursor = 0
        let bytesCount = weightsData.count

        func readShort() -> UInt16 {
            guard cursor + 2 <= bytesCount else { return 0 }
            let val = weightsData.withUnsafeBytes { ptr -> UInt16 in
                ptr.loadUnaligned(fromByteOffset: cursor, as: UInt16.self)
            }
            cursor += 2
            return UInt16(littleEndian: val)
        }

        func readFloat() -> Float {
            guard cursor + 4 <= bytesCount else { return 0.0 }
            let u32 = weightsData.withUnsafeBytes { ptr -> UInt32 in
                ptr.loadUnaligned(fromByteOffset: cursor, as: UInt32.self)
            }
            cursor += 4
            let val = UInt32(littleEndian: u32)
            return Float(bitPattern: val)
        }

        // 1. Read Primary W [48, pVocabSize]
        let pVocabSize = vocabObj["primary"]?.count ?? 20000
        var primaryW = [[Float]](repeating: [Float](repeating: 0.0, count: pVocabSize), count: 48)
        for comp in 0..<48 {
            for termIdx in 0..<pVocabSize {
                primaryW[comp][termIdx] = Float16Helper.halfToFloat(readShort())
            }
        }

        // 2. Read Channel W matrices
        var channelW: [String: [[Float]]] = [:]
        for chName in vocabObj.keys.sorted() {
            if chName == "primary" { continue }
            let chVocabSize = vocabObj[chName]?.count ?? 0
            var chMat = [[Float]](repeating: [Float](repeating: 0.0, count: chVocabSize), count: 48)
            for comp in 0..<48 {
                for termIdx in 0..<chVocabSize {
                    chMat[comp][termIdx] = Float16Helper.halfToFloat(readShort())
                }
            }
            channelW[chName] = chMat
        }

        // 3. Read Global PCA
        var globalPcaComp: [[Float]]? = nil
        var globalPcaMean: [Float]? = nil
        if hasGlobalPca && pcaDim > 0 {
            let inDim = 48 * (1 + channelW.count) + numericColumns.count
            var compMat = [[Float]](repeating: [Float](repeating: 0.0, count: inDim), count: pcaDim)
            for d in 0..<pcaDim {
                for i in 0..<inDim {
                    compMat[d][i] = Float16Helper.halfToFloat(readShort())
                }
            }
            globalPcaComp = compMat

            var meanVec = [Float](repeating: 0.0, count: inDim)
            for i in 0..<inDim {
                meanVec[i] = readFloat()
            }
            globalPcaMean = meanVec
        }

        return JourneyVectorizer(
            vocabularies: vocabObj,
            channelWeights: channelWeights,
            numericBlockWeight: numericBlockWeight,
            numericColumns: numericColumns,
            numericLog1pColumns: numericLog1pColumns,
            numericMean: numericMean,
            numericScale: numericScale,
            primaryW: primaryW,
            channelW: channelW,
            globalPcaComp: globalPcaComp,
            globalPcaMean: globalPcaMean
        )
    }
}
