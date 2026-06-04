//! A small, seedable random-number generator.
//!
//! Measurement needs uniform `f64`s in `[0, 1)`. We hand-roll a `xoshiro256++`
//! generator seeded through `SplitMix64` rather than pull in the `rand` crate,
//! keeping the simulator dependency-free. The generator is deterministic given
//! its seed, which makes measurement-based tests reproducible.

/// A deterministic pseudo-random generator (`xoshiro256++`).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seeds a generator. Any seed (including `0`) yields a valid stream; the
    /// seed is expanded through `SplitMix64` so the four words are well mixed.
    #[must_use]
    pub fn seed_from_u64(mut seed: u64) -> Self {
        // splitmix64: a fast mixer used to initialize the larger state.
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    /// Returns the next raw 64-bit value and advances the state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xoshiro256++: rotate-add output, then advance the linear state.
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Returns a uniform `f64` in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // take the top 53 bits, the f64 mantissa width, for an exact dyadic.
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::seed_from_u64(42);
        let mut b = Rng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::seed_from_u64(1);
        let mut b = Rng::seed_from_u64(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = Rng::seed_from_u64(7);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn f64_mean_is_near_half() {
        let mut r = Rng::seed_from_u64(123);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| r.next_f64()).sum::<f64>() / f64::from(n);
        // generous bound; just catches gross bias.
        assert!((mean - 0.5).abs() < 0.01);
    }
}
