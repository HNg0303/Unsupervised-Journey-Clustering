import Foundation

/// Post Processor mapping cluster IDs to business archetypes and enriching journeys with taxonomy metadata.
public final class JourneyPostProcessor: @unchecked Sendable {
    private let clusterMapping: ClusterMapping
    private let thresholds: ScoreThresholds
    private let minJourneyLength: Int

    public init(
        clusterMapping: ClusterMapping,
        thresholds: ScoreThresholds = ScoreThresholds(),
        minJourneyLength: Int = 4
    ) {
        self.clusterMapping = clusterMapping
        self.thresholds = thresholds
        self.minJourneyLength = minJourneyLength
    }

    /// Post-process an unscored collecting journey (not enough events).
    public func postProcessCollecting(prepared: PreparedJourney, provisional: Bool) -> MobilePrediction {
        return MobilePrediction(
            journeyId: prepared.journeyId,
            sessionId: prepared.sessionId,
            deviceId: prepared.deviceId,
            customerId: prepared.customerId,
            platform: prepared.platform,
            state: "collecting",
            boundaryReason: prepared.boundaryReason,
            eventsSeen: prepared.rawEvents.count,
            eventSequence: prepared.tokens,
            cluster: nil,
            clusterName: nil,
            businessFamily: nil,
            businessSubmodule: nil,
            businessDetail: nil,
            namingConfidence: nil,
            distanceToCentroid: nil,
            distanceLimit: thresholds.distanceP95,
            markovLogprob: nil,
            geometricAnomaly: false,
            generativeAnomaly: false,
            severeAnomaly: false,
            frictionFlags: "",
            nextAction: nil,
            nextActionShare: nil,
            medoidPath: nil
        )
    }

    /// Post-process a scored journey by enriching with cluster metadata, business taxonomy, and anomaly flags.
    public func postProcess(
        prepared: PreparedJourney,
        scoring: JourneyScoringOutput,
        provisional: Bool
    ) -> MobilePrediction {
        let state = provisional ? "provisional" : "final"

        let archetype = clusterMapping.lookup(clusterId: scoring.assignedClusterId)
            ?? clusterMapping.lookup(clusterId: scoring.nearestCentroidClusterId)

        let fallbackName: String
        if scoring.geometricAnomaly {
            fallbackName = "Chưa phân loại | Journey hỗn hợp/nhiễu"
        } else {
            fallbackName = "Archetype \(scoring.nearestCentroidClusterId)"
        }

        let clusterName = archetype?.clusterName ?? fallbackName
        let joinedFlags = scoring.frictionFlags.joined(separator: "|")

        return MobilePrediction(
            journeyId: prepared.journeyId,
            sessionId: prepared.sessionId,
            deviceId: prepared.deviceId,
            customerId: prepared.customerId,
            platform: prepared.platform,
            state: state,
            boundaryReason: prepared.boundaryReason,
            eventsSeen: prepared.rawEvents.count,
            eventSequence: prepared.tokens,
            cluster: scoring.assignedClusterId,
            clusterName: clusterName,
            businessFamily: archetype?.businessFamily,
            businessSubmodule: archetype?.businessSubmodule,
            businessDetail: archetype?.businessDetail,
            namingConfidence: archetype?.namingConfidence,
            distanceToCentroid: scoring.distanceToCentroid,
            distanceLimit: thresholds.distanceP95,
            markovLogprob: scoring.markovLogprob,
            geometricAnomaly: scoring.geometricAnomaly,
            generativeAnomaly: scoring.generativeAnomaly,
            severeAnomaly: scoring.severeAnomaly,
            frictionFlags: joinedFlags,
            nextAction: scoring.nextAction?.token,
            nextActionShare: scoring.nextAction?.observedShare,
            medoidPath: archetype?.medoidPath
        )
    }
}
