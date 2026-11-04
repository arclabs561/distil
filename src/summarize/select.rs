//! Budgeted selection over candidate sentences, shared by
//! [`SumBasic`](super::SumBasic), [`KlSum`](super::KlSum) and
//! [`Mmr`](super::Mmr).
//!
//! The functions work on indices `0..n` and never look at a document, so
//! candidates can come from anywhere (several documents, generated
//! sentences): the caller supplies relevance scores, a similarity matrix
//! and, for a [`Budget::Words`] budget, each candidate's length in words.
//! Results come in selection order with the criterion value at selection.

use super::Budget;

/// Pick candidates one at a time: each step takes the remaining candidate
/// that fits the budget with the highest `gain(i, selected)`, the earlier
/// one on ties (gains within a relative 1e-12, so that rounding does not
/// decide; NaN counts as the lowest gain), and stops when nothing fits.
/// `lengths[i]` (words) is used only by [`Budget::Words`]; missing lengths
/// count as 0.
pub fn greedy(
    n: usize,
    lengths: &[usize],
    budget: Budget,
    mut gain: impl FnMut(usize, &[usize]) -> f64,
) -> Vec<(usize, f64)> {
    let len = |i: usize| lengths.get(i).copied().unwrap_or(0);
    let mut left: Vec<usize> = (0..n).collect();
    let mut picked: Vec<usize> = Vec::new();
    let mut used = 0;
    let mut out = Vec::new();
    loop {
        let mut best: Option<(usize, f64)> = None;
        for (k, &i) in left.iter().enumerate() {
            if !budget.fits(picked.len(), used, len(i)) {
                continue;
            }
            let g = gain(i, &picked);
            let better = |b: f64| g > b && (b.is_infinite() || g - b > 1e-12 * b.abs().max(1.0));
            if best.is_none_or(|(_, b)| better(b)) {
                best = Some((k, if g.is_nan() { f64::NEG_INFINITY } else { g }));
            }
        }
        let Some((k, g)) = best else { break };
        let i = left.remove(k);
        used += len(i);
        picked.push(i);
        out.push((i, g));
    }
    out
}

/// Maximal marginal relevance (Carbonell and Goldstein 1998): each step
/// takes the candidate maximizing
///
/// `lambda * relevance[i] - (1 - lambda) * max over selected j of similarity[i][j]`
///
/// (the maximum is 0 before anything is selected).
pub fn mmr(
    relevance: &[f64],
    similarity: &[Vec<f64>],
    lambda: f64,
    lengths: &[usize],
    budget: Budget,
) -> Vec<(usize, f64)> {
    greedy(relevance.len(), lengths, budget, |i, picked| {
        let redundancy = picked.iter().map(|&j| similarity[i][j]).fold(0.0, f64::max);
        lambda * relevance[i] - (1.0 - lambda) * redundancy
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relevance 1.0, 0.9, 0.5; candidates 0 and 1 are near duplicates
    /// (similarity 0.95). With lambda 0.5: first 0 (0.5); then 1 scores
    /// 0.45 - 0.475 = -0.025 and 2 scores 0.25 - 0.05 = 0.2, so 2 is next.
    #[test]
    fn mmr_skips_near_duplicates() {
        let sim = vec![
            vec![1.0, 0.95, 0.1],
            vec![0.95, 1.0, 0.1],
            vec![0.1, 0.1, 1.0],
        ];
        let got = mmr(&[1.0, 0.9, 0.5], &sim, 0.5, &[], Budget::Sentences(3));
        let order: Vec<usize> = got.iter().map(|x| x.0).collect();
        assert_eq!(order, vec![0, 2, 1]);
        assert!((got[1].1 - 0.2).abs() < 1e-12 && (got[2].1 + 0.025).abs() < 1e-12);
        let two = mmr(&[1.0, 0.9, 0.5], &sim, 0.5, &[], Budget::Sentences(2));
        assert_eq!(two.len(), 2);
    }

    #[test]
    fn word_budget_skips_what_does_not_fit() {
        let got = greedy(3, &[5, 3, 2], Budget::Words(5), |i, _| [3.0, 2.0, 1.0][i]);
        let order: Vec<usize> = got.iter().map(|x| x.0).collect();
        assert_eq!(order, vec![0]);
        let got = greedy(3, &[6, 3, 2], Budget::Words(5), |i, _| [3.0, 2.0, 1.0][i]);
        let order: Vec<usize> = got.iter().map(|x| x.0).collect();
        assert_eq!(order, vec![1, 2]);
    }
}
