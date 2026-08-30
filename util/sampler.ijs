NB. util/sampler.ijs — Generic LLM sampling utilities
NB. Provides temperature scaling, top-k, top-p, min-p, weighted sampling
NB.
NB. Convention:
NB.   Dyadic verbs: `param sampler_foo data`   (x=param, y=data)
NB.   Monadic verbs: `sampler_foo data`
NB.
NB. In 4:0 definitions: x=left arg (param), y=right arg (data)
NB. Internal dyadic calls: use `(x& sampler_foo) y` form

NB. ----------------------------------------------------------------
NB. Temperature scaling — dyadic
NB. x = temperature scalar, y = logit array
NB. Scales logits:  y % (temp ^ 1)
NB. If temp = 0, returns y unchanged
NB. ----------------------------------------------------------------
coclass 'inference'
sampler_temp =: 4 : 0
  if. x = 0 do. y return. end.
  y % (x ^ 1)
)

NB. ----------------------------------------------------------------
NB. Softmax — monadic, numerically stable
NB. y = logit array
NB. ----------------------------------------------------------------
sampler_softmax =: 3 : 0
  max_x =. >./ y
  shifted =. y - max_x
  
  allsame =. *./ shifted = max_x
  
  if. allsame do.
    e =. 1 #~ # y
  else.
    e =. 2 ^ shifted
  end.
  
  denom =. +/ e
  e % denom
)

NB. ----------------------------------------------------------------
NB. Top-k indices — dyadic
NB. x = k scalar integer, y = logit array
NB. Returns indices of top-k logits in descending order
NB. ----------------------------------------------------------------
sampler_topk_indices =: 4 : 0
  k =. x
  if. 0 = k do. $0 return. end.
  sorted_idx =. \: y
  if. k > # y do. k =. # y end.
  k {. sorted_idx
)

NB. ----------------------------------------------------------------
NB. Top-k filter — dyadic (threshold-based)
NB. x = k scalar integer, y = logit array
NB. Filters to keep only the k largest logits
NB. Uses threshold: keep logits >= kth_largest
NB. ----------------------------------------------------------------
sampler_topk =: 4 : 0
  if. x = 0 do. y return. end.  NB. k=0 means no filtering
  kth =. {: x {. \:~ y
  NB. Keep logits >= kth; set excluded to -1e30 (NOT 0) so softmax's
  NB. 2^(shifted) underflows to ~0. Zeroing to 0 floods the softmax
  NB. denominator with 2^(0-max) for every excluded token.
  mask =. -. y < kth
  (mask * y) + ((-. mask) * _1e30)
)

NB. ----------------------------------------------------------------
NB. Top-p (nucleus) filter — dyadic
NB. Returns mask array same shape as y, 1 for keep, 0 for drop
NB. Works on probabilities (not logits)
NB. x = threshold (default 0.95), y = probability array
NB. ----------------------------------------------------------------
sampler_topp_mask =: 4 : 0
  probs =. y
  NB. x>0 -> x else default 0.95 (selection idiom; "If" without branching)
  threshold =. (0.95 , x) {~ 0 < x
  
  NB. Sort descending using grade down
  sorted =. \:~ probs
  
  NB. Cumulative sum using +/\ (prefix sum scan)
  cumsum =. +/\ sorted
  
  NB. Find first index where cumsum > threshold
  cuts =. I. threshold < cumsum
  if. 0 = # cuts do. (# probs) $ 1 return. end.  NB. keep all
  
  NB. Number of elements to keep (include the crossing element)
  keep_n =. >: {. cuts
  
  NB. Compute inverse permutation: orig_idx → sorted_pos
  perm =. \: probs  NB. sorted_pos → orig_idx
  inv_perm =. perm i.!.0 i. # probs  NB. orig_idx → sorted_pos (intolerant)
  
  NB. orig_mask[i] = 1 if sorted_pos < keep_n
  inv_perm < keep_n
)

NB. Apply top-p: zero out probabilities not in nucleus
sampler_topp =: 4 : 0
  mask =. (x& sampler_topp_mask) y
  mask * y
)

NB. ----------------------------------------------------------------
NB. Min-p filter — dyadic
NB. x = min_p scalar (default 0.0), y = probability array
NB. Filters: prob < min_p x max_prob
NB. ----------------------------------------------------------------
sampler_minp_mask =: 4 : 0
  probs =. y
  NB. x>0 -> x else default 0.0 (selection idiom)
  min_p =. (0.0 , x) {~ 0 < x
  
  if. min_p = 0 do. (# probs) $ 1 return. end.  NB. disabled
  
  max_prob =. >./ probs
  threshold =. min_p * max_prob
  probs > threshold
)

NB. Apply min-p: zero out probabilities below threshold
sampler_minp =: 4 : 0
  mask =. (x& sampler_minp_mask) y
  mask * y
)

NB. ----------------------------------------------------------------
NB. Weighted random sample — monadic
NB. y = probability array (must sum to ~1)
NB. Returns a scalar index in 0..(#y)-1
NB. Uses cumulative probs + uniform random in (0,1)
NB. Uses large integer random divided by 1000000.0 for float in [0,1)
NB. ----------------------------------------------------------------
sampler_weighted_sample =: 3 : 0
  cum =. +/\ y
  r =. (? 1000000) % 1000000.0
  {. I. -. r > cum
)

NB. ----------------------------------------------------------------
NB. Full sampler pipeline — dyadic
NB. x = parameters array <temp; k; p; min_p>
NB. y = logit array
NB. Returns: scalar token index
NB. Default params: <1.0; 0; 0.95; 0.0> (k=0 means disabled)
NB. ----------------------------------------------------------------
sampler_sample =: 4 : 0
  params =. x
  
  NB. Parse parameters — params may be single or double boxed
  NB. Handle both: <temp;k;p;min_p> and <<temp;k;p;min_p>>
  if. 1 = # params do.
    flat =. > > params   NB. double unbox
  else.
    flat =. > params     NB. single unbox
  end.
  temp =. 1.0
  k =. 0
  p =. 0.95
  min_p =. 0.0
  
  if. 0 < # flat do. temp =. 0 { flat end.
  if. 1 < # flat do. k =. 1 { flat end.
  if. 2 < # flat do. p =. 2 { flat end.
  if. 3 < # flat do. min_p =. 3 { flat end.

  NB. Greedy shortcut: temp=0 picks the argmax of the RAW logits. All the
  NB. downstream filters (top-k, softmax, top-p, min-p) are monotonic and
  NB. always retain the argmax, so skipping them is exact — and it removes a
  NB. full-vocab softmax + top-p sort per generated token (~0.008s/token vs
  NB. ~2.4e-5s for the argmax on a 248k-vocab model).
  if. temp = 0 do. y i. >./ y return. end.

  NB. Step 1: Temperature scaling
  if. temp > 0 do.
    scaled =. temp sampler_temp y
  else.
    scaled =. y
  end.
  
  NB. Step 2: Top-k
  if. 0 < k do.
    logits =. k sampler_topk scaled
  else.
    logits =. scaled
  end.
  
  NB. Step 3: Softmax → probabilities
  probs =. sampler_softmax logits
  
  NB. Step 4: Top-p
  probs =. p sampler_topp probs
  
  NB. Step 5: Min-p
  probs =. min_p sampler_minp probs
  
  NB. Step 6: Renormalize after filters
  prob_sum =. +/ probs
  if. prob_sum = 0 do.
    probs =. sampler_softmax y  NB. fallback: use original logits
  else.
    probs =. probs % prob_sum
  end.
  
  NB. Step 7: Weighted random sample or greedy (argmax)
  if. temp = 0 do.
    sampler_greedy probs
  else.
    sampler_weighted_sample probs
  end.
)

NB. ----------------------------------------------------------------
NB. Greedy sampling (argmax) — monadic
NB. Returns first index of max value
NB. ----------------------------------------------------------------
sampler_greedy =: 3 : 0
  y i. >./ y
)
