import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// Fast IEEE 754 half-precision float to 32-bit float converter and L2 normalization utilities.
public enum Float16Helper {
    /// Convert 16-bit IEEE 754 half-precision bit pattern to 32-bit Float.
    @inline(__always)
    public static func halfToFloat(_ hbits: UInt16) -> Float {
        #if arch(arm64) || arch(x86_64)
        return Float(Float16(bitPattern: hbits))
        #else
        let h = Int(hbits)
        var mant = h & 0x03ff
        let exp = h & 0x7c00
        let sign: Float = (h & 0x8000 != 0) ? -1.0 : 1.0

        if exp == 0x7c00 {
            return (mant != 0) ? Float.nan : sign * Float.infinity
        }
        if exp != 0 {
            let fExp = (exp >> 10) - 15 + 127
            let u32 = UInt32(((h & 0x8000) << 16) | (fExp << 23) | (mant << 13))
            return Float(bitPattern: u32)
        }
        if mant == 0 { return sign * 0.0 }
        // Subnormal
        while (mant & 0x0400) == 0 {
            mant = mant << 1
        }
        mant = mant & ~0x03ff
        let fExp = 1 - 15 + 127
        let u32 = UInt32(((h & 0x8000) << 16) | (fExp << 23) | (mant << 13))
        return Float(bitPattern: u32)
        #endif
    }

    /// Convert 16-bit signed integer buffer to 32-bit Float.
    @inline(__always)
    public static func halfToFloat(_ hbits: Int16) -> Float {
        return halfToFloat(UInt16(bitPattern: hbits))
    }

    /// L2-normalizes an array of Floats in-place.
    public static func l2Normalize(_ values: inout [Float]) {
        guard !values.isEmpty else { return }
        #if canImport(Accelerate)
        var sumSq: Float = 0.0
        vDSP_svesq(values, 1, &sumSq, vDSP_Length(values.count))
        let norm = sqrt(sumSq)
        if norm > 1e-12 {
            var divisor = norm
            vDSP_vsdiv(values, 1, &divisor, &values, 1, vDSP_Length(values.count))
        }
        #else
        var sumSq: Double = 0.0
        for v in values { sumSq += Double(v * v) }
        let norm = Float(sqrt(sumSq))
        if norm > 1e-12 {
            for i in 0..<values.count { values[i] /= norm }
        }
        #endif
    }

    /// L2-normalizes an array of Doubles in-place.
    public static func l2Normalize(_ values: inout [Double]) {
        guard !values.isEmpty else { return }
        var sumSq: Double = 0.0
        for v in values { sumSq += (v * v) }
        let norm = sqrt(sumSq)
        if norm > 1e-12 {
            for i in 0..<values.count { values[i] /= norm }
        }
    }
}
