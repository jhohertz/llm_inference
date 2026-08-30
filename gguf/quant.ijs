NB. ================================================================
NB. Quantization decoders - GGUF tensor etypes -> native J float64.
NB. Each verb: y = <ne; <slice>> (same interface as f16_load), returns
NB. an (ne,) float64 array. All vectorized over blocks (no per-element
NB. loops); index maps for the K/IQ types are precomputed once at load.
NB. Layouts/algos follow llama.cpp ggml-common.h + ggml-quants.c
NB. (dequantize_row_*). Tables come from gguf/quant_tables.ijs.
NB. ================================================================
coclass 'inference'
require 'llm/inference/gguf/quant_tables'

u8 =: a. i. ]   NB. char slice -> uint8 array

NB. ---- block layout by etype (ggml.h numbering) ----
NB. qblk_elems = elements per block; qblk_bytes = bytes per block.
NB. etypes 2 3 6 7 8 9 10..19 20 21..23 24..28 29
qblk_elems =: 32 32 32 32 32 32 256 256 256 256 256 256 256 256 256 256 32 256 256 256 1 1 1 1 1 256
qblk_bytes =: 18 20 22 24 34 36 84 110 144 176 210 292 66 74 98 50 18 110 82 136 1 2 4 8 8 56
NB. etype -> qblk index (2..29, 30=BF16); _1 = not a quant type
qblk_idx =: _1 _1 0 1 _1 _1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 _1 _1 _1 _1 _1 _1 _1 _1 _1 _1 _1 _1 _1

NB. ---- simple decoders (32-elem blocks) ----

q4_0_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 32
  rows =. (nb,18) $ s
  d =. f16_load (<nb), <(, (0 1 {"1 rows))
  qs =. u8 (2 + i.16) {"1 rows
  lo =. (qs 17 b. 15) - 8
  hi =. (<. qs % 16) - 8
  , ((lo ,"1 hi) * d)
)

q4_1_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 32
  rows =. (nb,20) $ s
  d =. f16_load (<nb), <(, (0 1 {"1 rows))
  m =. f16_load (<nb), <(, (2 3 {"1 rows))
  qs =. u8 (4 + i.16) {"1 rows
  lo =. qs 17 b. 15
  hi =. <. qs % 16
  , ((lo ,"1 hi) * d) + m
)

q5_0_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 32
  rows =. (nb,22) $ s
  d =. f16_load (<nb), <(, (0 1 {"1 rows))
  qh =. (_2) 3!:4 , (2 + i.4) {"1 rows
  qs =. u8 (6 + i.16) {"1 rows
  j =. i.16
  xh0 =. ((<. qh (%/) 2 ^ j) * 16) 17 b. 16
  xh1 =. (<. qh (%/) 2 ^ (12 + j)) 17 b. 16
  lo =. (qs 17 b. 15) 23 b. xh0
  hi =. (<. qs % 16) 23 b. xh1
  , ((lo ,"1 hi) - 16) * d
)

q5_1_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 32
  rows =. (nb,24) $ s
  d =. f16_load (<nb), <(, (0 1 {"1 rows))
  m =. f16_load (<nb), <(, (2 3 {"1 rows))
  qh =. (_2) 3!:4 , (4 + i.4) {"1 rows
  qs =. u8 (8 + i.16) {"1 rows
  j =. i.16
  xh0 =. ((<. qh (%/) 2 ^ j) * 16) 17 b. 16
  xh1 =. (<. qh (%/) 2 ^ (12 + j)) 17 b. 16
  lo =. (qs 17 b. 15) 23 b. xh0
  hi =. (<. qs % 16) 23 b. xh1
  , ((lo ,"1 hi) * d) + m
)

q8_0_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 32
  rows =. (nb,34) $ s
  d =. f16_load (<nb), <(, (0 1 {"1 rows))
  qs =. u8 (2 + i.32) {"1 rows
  qs =. qs - 256 * qs > 127
  , (qs * d)
)

iq4_nl_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 32
  rows =. (nb,18) $ s
  d =. f16_load (<nb), <(, (0 1 {"1 rows))
  qs =. u8 (2 + i.16) {"1 rows
  lo =. kvalues_iq4nl {~ qs 17 b. 15
  hi =. kvalues_iq4nl {~ <. qs % 16
  , ((lo ,"1 hi) * d)
)

NB. ---- native integer / float types ----

i8_decode =: 3 : 0
  ne =. > 0 { y
  b =. u8 > 1 { y
  , (b - 256 * b > 127)
)

i16_decode =: 3 : 0
  ne =. > 0 { y
  , (_1) 3!:4 > 1 { y
)

i32_decode =: 3 : 0
  ne =. > 0 { y
  , (_2) 3!:4 > 1 { y
)

u32_decode =: 3 : 0
  v =. _2 (3!:4) y
  v + 4294967296 * v < 0
)

i64_decode =: 3 : 0
  ne =. > 0 { y
  , (_3) 3!:4 > 1 { y
)

NB. ---- K-quant super-block index maps (positions e = 0..255) ----
NB. Built once at load; each map is a (256,) integer vector.
q2k_build =: 3 : 0
  e =. i. 256
  h =. <. e % 128
  p =. e - 128 * h
  j =. <. p % 32
  l =. p - 32 * j
  l16 =. l >: 16
  lmod =. l - 16 * l16
  si =. (8 * h) + (2 * j) + l16
  b =. (32 * h) + lmod + (16 * l16)
  sh =. 2 * j
  (<si) , (<b) , (<sh)
)
q2k_map =: q2k_build ''

q3k_build =: 3 : 0
  e =. i. 256
  h =. <. e % 128
  p =. e - 128 * h
  j =. <. p % 32
  l =. p - 32 * j
  l16 =. l >: 16
  lmod =. l - 16 * l16
  si =. (8 * h) + (2 * j) + l16
  b =. (32 * h) + lmod + (16 * l16)
  sh =. 2 * j
  hmb =. lmod + (16 * l16)   NB. hm byte = l (0..31), no half offset
  m =. 2 ^ j + 4 * h         NB. mask bit = 2^(j+4h), persists across halves
  (<si) , (<b) , (<sh) , (<hmb) , (<m)
)
q3k_map =: q3k_build ''

q4k_build =: 3 : 0
  e =. i. 256
  g =. <. e % 32
  l =. e - 32 * g
  j =. <. g % 2
  base =. 32 * j
  even =. 0 = 2 | g
  (<g) , (<l) , (<base) , (<even)
)
q4k_map =: q4k_build ''

q5k_map =: q4k_map   NB. same layout; adds qh byte + u1/u2

q6k_build =: 3 : 0
  e =. i. 256
  h =. <. e % 128
  p =. e - 128 * h
  q =. <. p % 32
  l =. p - 32 * q
  is =. <. l % 16
  sc_idx =. (8 * h) + is + (2 * q)
  ql_byte =. (64 * h) + l + (32 * (1 = 2 | q))
  qh_byte =. (32 * h) + l
  qh_shift =. 2 * q
  (<sc_idx) , (<ql_byte) , (<qh_byte) , (<qh_shift)
)
q6k_map =: q6k_build ''

q2_K_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,84) $ s
  scales =. u8 (0 + i.16) {"1 rows
  qs =. u8 (16 + i.64) {"1 rows
  d =. f16_load (<nb), <(, (80 + i.2) {"1 rows)
  dmin =. f16_load (<nb), <(, (82 + i.2) {"1 rows)
  si =. > 0 { q2k_map
  b =. > 1 { q2k_map
  sh =. > 2 { q2k_map
  sc =. si {"1 scales
  qv =. b {"1 qs
  dl =. d * (sc 17 b. 15)
  ml =. dmin * (<. sc % 16)
  v =. (<. qv %"1 (2 ^ sh)) 17 b. 3
  , ((dl * v) - ml)
)

q3_K_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,110) $ s
  hm =. u8 (0 + i.32) {"1 rows
  qs =. u8 (32 + i.64) {"1 rows
  sc12 =. (96 + i.12) {"1 rows   NB. raw chars for 3!:4
  d =. f16_load (<nb), <(, (108 + i.2) {"1 rows)
  NB. aux reorder: 12 bytes -> 3 uint32 (unsigned) -> 4 uint32 -> 16 signed bytes
  c12 =. , sc12
  u =. (nb,3) $ u32_decode c12
  a0 =. 0 { |: u
  a1 =. 1 { |: u
  a2 =. 2 { |: u
  mask1 =. 50529027            NB. 0x03030303
  mask2 =. 252645135           NB. 0x0f0f0f0f
  aux0 =. (a0 17 b. mask2) 23 b. ((a2 17 b. mask1) * 16)
  aux1 =. (a1 17 b. mask2) 23 b. ((<. a2 % 4) 17 b. mask1) * 16
  aux2 =. ((<. a0 % 16) 17 b. mask2) 23 b. ((<. a2 % 16) 17 b. mask1) * 16
  aux3 =. ((<. a1 % 16) 17 b. mask2) 23 b. ((<. a2 % 64) 17 b. mask1) * 16
  NB. aux bytes -> 16 signed scales (block-major: per-block aux0..aux3)
  aux_ch =. 2 (3!:4) , (aux0 ,. aux1 ,. aux2 ,. aux3)
  scales =. (nb,16) $ u8 aux_ch
  scales =. scales - 256 * scales > 127
  si =. > 0 { q3k_map
  b =. > 1 { q3k_map
  sh =. > 2 { q3k_map
  hmb =. > 3 { q3k_map
  m =. > 4 { q3k_map
  sc =. si {"1 scales
  qv =. b {"1 qs
  hmbv =. hmb {"1 hm
  dl =. d * (sc - 32)
  v =. (<. qv %"1 (2 ^ sh)) 17 b. 3
  v =. v - 4 * (0 = hmbv 17 b."1 m)
  , dl * v
)

q4_K_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,144) $ s
  qs =. u8 (16 + i.128) {"1 rows
  sc12 =. u8 (4 + i.12) {"1 rows
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  dmin =. f16_load (<nb), <(, (2 + i.2) {"1 rows)
  g =. > 0 { q4k_map
  l =. > 1 { q4k_map
  base =. > 2 { q4k_map
  even =. > 3 { q4k_map
  NB. get_scale_min_k4(g): g<4: d=sc[g]&63, m=sc[g+4]&63
  NB. g>=4: d=(sc[g+4]&0xF)|((sc[g-4]>>6)<<4), m=(sc[g+4]>>4)|((sc[g]>>6)<<4)
  gs =. g < 4
  scg =. g {"1 sc12
  scg4 =. (g + 4) {"1 sc12
  scgm4 =. (g - 4) {"1 sc12
  sd =. (scg 17 b. 63) *"1 gs
  sd =. sd + ((scg4 17 b. 15) 23 b. ((<. scgm4 % 64) * 16)) *"1 -. gs
  sm =. (scg4 17 b. 63) *"1 gs
  sm =. sm + ((<. scg4 % 16) 23 b. ((<. scg % 64) * 16)) *"1 -. gs
  qb =. (base + l) {"1 qs
  v =. (qb 17 b. 15) *"1 even
  v =. v + (<. qb % 16) *"1 -. even
  , ((d * sd * v) - dmin * sm)
)

q5_K_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,176) $ s
  ql =. u8 (48 + i.128) {"1 rows
  qh =. u8 (16 + i.32) {"1 rows
  sc12 =. u8 (4 + i.12) {"1 rows
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  dmin =. f16_load (<nb), <(, (2 + i.2) {"1 rows)
  g =. > 0 { q5k_map
  l =. > 1 { q5k_map
  base =. > 2 { q5k_map
  even =. > 3 { q5k_map
  j =. <. g % 2
  u1 =. 2 ^ 2 * j
  u2 =. 2 ^ 1 + 2 * j
  gs =. g < 4
  scg =. g {"1 sc12
  scg4 =. (g + 4) {"1 sc12
  scgm4 =. (g - 4) {"1 sc12
  sd =. (scg 17 b. 63) *"1 gs
  sd =. sd + ((scg4 17 b. 15) 23 b. ((<. scgm4 % 64) * 16)) *"1 -. gs
  sm =. (scg4 17 b. 63) *"1 gs
  sm =. sm + ((<. scg4 % 16) 23 b. ((<. scg % 64) * 16)) *"1 -. gs
  qlb =. (base + l) {"1 ql
  qhb =. l {"1 qh
  v =. (qlb 17 b. 15) *"1 even
  v =. v + (<. qlb % 16) *"1 -. even
  hi =. 16 * (0 ~: (qhb 17 b."1 u1)) *"1 even
  hi =. hi + 16 * (0 ~: (qhb 17 b."1 u2)) *"1 -. even
  , ((d * sd * (v + hi)) - dmin * sm)
)

q6_K_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,210) $ s
  ql =. u8 (0 + i.128) {"1 rows
  qh =. u8 (128 + i.64) {"1 rows
  sc =. u8 (192 + i.16) {"1 rows
  d =. f16_load (<nb), <(, (208 + i.2) {"1 rows)
  sc_idx =. > 0 { q6k_map
  ql_byte =. > 1 { q6k_map
  qh_byte =. > 2 { q6k_map
  qh_shift =. > 3 { q6k_map
  q =. <. qh_shift % 2
  hiq =. q >: 2
  ssc =. sc_idx {"1 sc
  ssc =. ssc - 256 * ssc > 127
  qlb =. ql_byte {"1 ql
  qhb =. qh_byte {"1 qh
  nib =. (qlb 17 b. 15) *"1 -. hiq
  nib =. nib + ((<. qlb % 16) 17 b. 15) *"1 hiq
  hb =. (<. qhb %"1 (2 ^ qh_shift)) 17 b. 3
  v =. (nib 23 b. hb * 16) - 32
  , d * ssc * v
)

q8_K_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,292) $ s
  d =. f32_decode (<nb), <(, (i.4) {"1 rows)
  qs =. u8 (4 + i.256) {"1 rows
  qs =. qs - 256 * qs > 127
  , (qs * d)
)

NB. ---- IQ index maps (positions e = 0..255) ----
NB. qdiag: pick the diagonal of a (nb,256,8) array: for each position e,
NB. select byte jm[e]. y = <arr3 ; jm>
qdiag =: 3 : 0
  g =. > 0 { y
  j =. > 1 { y
  nb =. 0 { $ g
  (nb,256) $ (, g) {~ ((i. nb * 256) * 8) + (nb * 256) $ j
)
iq2s_maps =: 3 : 0
  e =. i. 256
  ib32 =. <. e % 32
  l =. <. (e - 32 * ib32) % 8
  j =. (e - 32 * ib32) - (8 * l)
  gi =. (4 * ib32) + l        NB. qs byte index (iq2_xxs aux byte l)
  ibm =. ib32
  lm =. l
  jm =. j
  (<gi) , (<ibm) , (<lm) , (<jm)
)
iq2xxs_map =: iq2s_maps ''

iq2x_map =: 3 : 0
  e =. i. 256
  ib32 =. <. e % 32
  l =. <. (e - 32 * ib32) % 8
  j =. (e - 32 * ib32) - (8 * l)
  qi =. (4 * ib32) + l        NB. qs16 index
  ibm =. ib32
  lm =. l
  jm =. j
  (<qi) , (<ibm) , (<lm) , (<jm)
)
iq2xs_map =: iq2x_map ''

iq2s_idxmap =: 3 : 0
  e =. i. 256
  ib32 =. <. e % 32
  l =. <. (e - 32 * ib32) % 8
  j =. (e - 32 * ib32) - (8 * l)
  qi =. (4 * ib32) + l        NB. qs byte index (grid low byte)
  qhi =. ib32                 NB. qh byte index
  sgi =. (4 * ib32) + l       NB. signs byte index (qs+32)
  ibm =. ib32
  lm =. l
  jm =. j
  (<qi) , (<qhi) , (<sgi) , (<ibm) , (<lm) , (<jm)
)
iq2s_map =: iq2s_idxmap ''

iq3x_map =: 3 : 0
  e =. i. 256
  ib32 =. <. e % 32
  l =. <. (e - 32 * ib32) % 8
  j =. (e - 32 * ib32) - (8 * l)
  g1i =. (8 * ib32) + (2 * l)
  g2i =. g1i + 1
  ibm =. ib32
  lm =. l
  jm =. j
  (<g1i) , (<g2i) , (<ibm) , (<lm) , (<jm)
)
iq3xxs_map =: iq3x_map ''

iq3s_map =: 3 : 0
  e =. i. 256
  p =. <. e % 64
  e2 =. e - 64 * p
  half =. <. e2 % 32
  l =. <. (e2 - 32 * half) % 8
  j =. (e2 - 32 * half) - (8 * l)
  q1i =. (16 * p) + (8 * half) + (2 * l)
  q2i =. q1i + 1
  qhi =. (2 * p) + half
  sgi =. (8 * p) + (4 * half) + l
  pm =. p
  hm =. half
  lm =. l
  jm =. j
  (<q1i) , (<q2i) , (<qhi) , (<sgi) , (<pm) , (<hm) , (<lm) , (<jm)
)
iq3s_idxmap =: iq3s_map ''

iq1s_map =: 3 : 0
  e =. i. 256
  ib =. <. e % 32
  l =. <. (e - 32 * ib) % 8
  j =. (e - 32 * ib) - (8 * l)
  qi =. (4 * ib) + l
  ibm =. ib
  lm =. l
  jm =. j
  (<qi) , (<ibm) , (<lm) , (<jm)
)
iq1s_idxmap =: iq1s_map ''

iq1m_map =: 3 : 0
  e =. i. 256
  ib =. <. e % 32
  l =. <. (e - 32 * ib) % 8
  j =. (e - 32 * ib) - (8 * l)
  qi =. (4 * ib) + l
  qhi =. (2 * ib) + (l > 1)
  ibm =. ib
  lm =. l
  jm =. j
  (<qi) , (<qhi) , (<ibm) , (<lm) , (<jm)
)
iq1m_idxmap =: iq1m_map ''

NB. ---- IQ decoders ----

iq4_xs_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,136) $ s
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  scales_h =. 0 (3!:4) , (2 + i.2) {"1 rows        NB. uint16 (nb,)
  scales_l =. u8 (4 + i.4) {"1 rows                NB. (nb,4)
  qs =. u8 (8 + i.128) {"1 rows                    NB. (nb,128)
  e =. i. 256
  ib =. <. e % 32
  j =. e - 32 * ib
  sc_l_e =. (<. ib % 2) {"1 scales_l               NB. (nb,256)
  shl =. 4 * (1 = 2 | ib)                          NB. (256,)
  lsl =. (<. sc_l_e %"1 (2 ^ shl)) 17 b. 15        NB. (nb,256)
  sc_h_e =. scales_h */ (256 $ 1)                  NB. (nb,256)
  hsh =. (<. sc_h_e %"1 (2 ^ (2 * ib))) 17 b. 3    NB. (nb,256)
  ls =. lsl 23 b. (hsh * 16)                       NB. (nb,256)
  dl =. d * (ls - 32)                              NB. (nb,256)
  qi =. (16 * ib) + j - 16 * (j > 15)              NB. (256,)
  qb =. qi {"1 qs                                  NB. (nb,256)
  lo =. kvalues_iq4nl {~ qb 17 b. 15
  hi =. kvalues_iq4nl {~ <. qb % 16
  v =. (lo *"1 (j < 16)) + hi *"1 (j >: 16)        NB. (nb,256)
  , dl * v
)

iq2_xxs_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,66) $ s
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  qsc =. (2 + i.64) {"1 rows                       NB. chars (nb,64)
  qn =. u8 qsc                                     NB. (nb,64)
  NB. aux32[1] = uint32 at qs[4*ib32+4 .. 4*ib32+7] (8-byte group, stride 4, overlap)
  idx =. (4 + 4 * i.8) +/ i.4                      NB. (8,4): [4*ib32+4 .. 4*ib32+7]
  aux1 =. (nb,8) $ u32_decode , (((i. nb) * 64) +/ , idx) { , qsc   NB. (nb,8) uint32
  gi =. > 0 { iq2xxs_map                           NB. (256,) qs byte
  ibm =. > 1 { iq2xxs_map
  lm =. > 2 { iq2xxs_map
  jm =. > 3 { iq2xxs_map
  grid_idx =. gi {"1 qn                             NB. (nb,256)
  grid =. iq2xxs_grid {~ grid_idx                   NB. (nb,256,8)
  db0 =. d * (0.5 + (<. aux1 % 268435456)) * 0.25   NB. (nb,8) aux1>>28
  db_e =. ibm {"1 db0                 NB. (nb,256)
  si =. (<. (ibm {"1 aux1) %"1 (2 ^ 7 * lm)) 17 b. 127  NB. (nb,256)
  ksigns =. ksigns_iq2xs {~ si                     NB. (nb,256,8)
  sf =. 1 - 2 * (0 ~: (ksigns (17 b.)/ kmask_iq2xs))
  , db_e * (qdiag (<grid) , <jm) * (qdiag (<sf) , <jm)
)

iq2_xs_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,74) $ s
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  qs16 =. (nb,32) $ 0 (3!:4) , (2 + i.64) {"1 rows   NB. uint16 (nb,32)
  sc =. u8 (66 + i.8) {"1 rows                       NB. (nb,8)
  qi =. > 0 { iq2xs_map                             NB. (256,)
  ibm =. > 1 { iq2xs_map
  lm =. > 2 { iq2xs_map
  jm =. > 3 { iq2xs_map
  v =. qi {"1 qs16                                   NB. (nb,256)
  grid_idx =. v 17 b. 511
  signs_idx =. <. v % 512
  grid =. iq2xs_grid {~ grid_idx                     NB. (nb,256,8)
  ksigns =. ksigns_iq2xs {~ signs_idx                NB. (nb,256,8)
  db0 =. d * (0.5 + (sc 17 b. 15)) * 0.25            NB. (nb,8)
  db1 =. d * (0.5 + (<. sc % 16)) * 0.25             NB. (nb,8)
  db0_e =. ibm {"1 db0
  db1_e =. ibm {"1 db1
  NB. C: dl = db[l/2] (l 0,1 -> db0; l 2,3 -> db1), NOT parity
  dl =. (db0_e *"1 (lm < 2)) + db1_e *"1 (lm >: 2)
  sf =. 1 - 2 * (0 ~: (ksigns (17 b.)/ kmask_iq2xs))
  , dl * (qdiag (<grid) , <jm) * (qdiag (<sf) , <jm)
)

iq2_s_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,82) $ s
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  qn =. u8 (2 + i.64) {"1 rows                       NB. (nb,64)
  qh =. u8 (66 + i.8) {"1 rows                       NB. (nb,8)
  sc =. u8 (74 + i.8) {"1 rows                       NB. (nb,8)
  sg =. u8 (34 + i.32) {"1 rows                      NB. signs (nb,32)
  qi =. > 0 { iq2s_map                               NB. (256,)
  qhi =. > 1 { iq2s_map
  sgi =. > 2 { iq2s_map
  ibm =. > 3 { iq2s_map
  lm =. > 4 { iq2s_map
  jm =. > 5 { iq2s_map
  qb =. qi {"1 qn                                     NB. (nb,256)
  qhb =. qhi {"1 qh                                   NB. (nb,256)
  hval =. (qhb *"1 (2 ^ 8 - 2 * lm)) 17 b. 768
  grid_idx =. qb 23 b. hval                           NB. (nb,256)
  grid =. iq2s_grid {~ grid_idx                       NB. (nb,256,8)
  sge =. sgi {"1 sg                                   NB. (nb,256)
  sf =. 1 - 2 * (0 ~: (sge (17 b.)/ kmask_iq2xs))
  db0 =. d * (0.5 + (sc 17 b. 15)) * 0.25
  db1 =. d * (0.5 + (<. sc % 16)) * 0.25
  db0_e =. ibm {"1 db0
  db1_e =. ibm {"1 db1
  NB. C: dl = db[l/2] (l 0,1 -> db0; l 2,3 -> db1), NOT parity
  dl =. (db0_e *"1 (lm < 2)) + db1_e *"1 (lm >: 2)
  , dl * (qdiag ((<grid) , <jm)) * (qdiag ((<sf) , <jm))
)

iq3_xxs_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,98) $ s
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  qn =. u8 (2 + i.64) {"1 rows                        NB. (nb,64)
  sasc =. (66 + i.32) {"1 rows                        NB. chars (nb,32)
  sas8 =. (nb,8,4) $ sasc                             NB. chars [ib32, byte]
  aux =. (nb,8) $ u32_decode , sas8                   NB. (nb,8)
  g1i =. > 0 { iq3xxs_map
  g2i =. > 1 { iq3xxs_map
  ibm =. > 2 { iq3xxs_map
  lm =. > 3 { iq3xxs_map
  jm =. > 4 { iq3xxs_map
  grid1 =. iq3xxs_grid {~ (g1i {"1 qn)                NB. (nb,256,4)
  grid2 =. iq3xxs_grid {~ (g2i {"1 qn)                NB. (nb,256,4)
  db0 =. d * (0.5 + (<. aux % 268435456)) * 0.5       NB. (nb,8) aux>>28
  db_e =. ibm {"1 db0                   NB. (nb,256)
  si =. (<. (ibm {"1 aux) %"1 (2 ^ 7 * lm)) 17 b. 127  NB. (nb,256)
  ksigns =. ksigns_iq2xs {~ si                       NB. (nb,256,8)
  sf =. 1 - 2 * (0 ~: (ksigns (17 b.)/ kmask_iq2xs))
  sf1 =. (i.4) {"1 sf                                  NB. j 0..3
  sf2 =. (4 + i.4) {"1 sf                              NB. j 4..7
  out =. (grid1 * sf1) ,"1 (grid2 * sf2)              NB. (nb,256,8)
  , db_e * (qdiag (<out) , <jm)
)

iq3_s_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,110) $ s
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  qn =. u8 (2 + i.64) {"1 rows                        NB. (nb,64)
  qh =. u8 (66 + i.8) {"1 rows                        NB. (nb,8)
  sg =. u8 (74 + i.32) {"1 rows                       NB. (nb,32)
  scp =. u8 (106 + i.4) {"1 rows                      NB. (nb,4)
  q1i =. > 0 { iq3s_idxmap
  q2i =. > 1 { iq3s_idxmap
  qhi =. > 2 { iq3s_idxmap
  sgi =. > 3 { iq3s_idxmap
  pm =. > 4 { iq3s_idxmap
  hm =. > 5 { iq3s_idxmap
  lm =. > 6 { iq3s_idxmap
  jm =. > 7 { iq3s_idxmap
  qb1 =. q1i {"1 qn
  qb2 =. q2i {"1 qn
  qhb =. qhi {"1 qh                                    NB. (nb,256)
  h1 =. (qhb *"1 (2 ^ 8 - 2 * lm)) 17 b. 256
  h2 =. (qhb *"1 (2 ^ 7 - 2 * lm)) 17 b. 256
  grid1 =. iq3s_grid {~ (qb1 23 b. h1)                 NB. (nb,256,4)
  grid2 =. iq3s_grid {~ (qb2 23 b. h2)                 NB. (nb,256,4)
  sge =. sgi {"1 sg                                     NB. (nb,256)
  sf =. 1 - 2 * (0 ~: (sge (17 b.)/ kmask_iq2xs))
  sf1 =. (i.4) {"1 sf
  sf2 =. (4 + i.4) {"1 sf
  db1 =. d * (1 + 2 * (scp 17 b. 15))                   NB. (nb,4)
  db2 =. d * (1 + 2 * (<. scp % 16))                    NB. (nb,4)
  db1_e =. pm {"1 db1
  db2_e =. pm {"1 db2
  dl =. (db1_e *"1 (hm < 1)) + db2_e *"1 (hm >: 1)
  out =. (grid1 * sf1) ,"1 (grid2 * sf2)
  , dl * (qdiag (<out) , <jm)
)

iq1_s_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,50) $ s
  d =. f16_load (<nb), <(, (i.2) {"1 rows)
  qn =. u8 (2 + i.32) {"1 rows                        NB. (nb,32)
  qh16 =. (nb,8) $ 0 (3!:4) , (34 + i.16) {"1 rows     NB. uint16 (nb,8)
  qi =. > 0 { iq1s_idxmap
  ibm =. > 1 { iq1s_idxmap
  lm =. > 2 { iq1s_idxmap
  jm =. > 3 { iq1s_idxmap
  qb =. qi {"1 qn                                       NB. (nb,256)
  qhe =. ibm {"1 qh16                     NB. (nb,256)
  hval =. (<. qhe %"1 (2 ^ 3 * lm)) 17 b. 7             NB. (nb,256)
  idx =. qb 23 b. (hval * 256)                          NB. (nb,256)
  grid =. iq1s_grid {~ idx                              NB. (nb,256,8)
  dl =. d * (1 + 2 * ((<. qhe %"1 (2 ^ 12)) 17 b. 7))   NB. (nb,256)
  NB. delta: -0.125 if qh&0x8000 else +0.125
  delta =. 0.125 - 0.25 * (0 ~: qhe 17 b. 32768)
  , dl * ((qdiag ((<grid) , <jm)) + delta)
)

iq1_m_decode =: 3 : 0
  ne =. > 0 { y
  s =. > 1 { y
  nb =. ne % 256
  rows =. (nb,56) $ s
  qn =. u8 (0 + i.32) {"1 rows                        NB. (nb,32) qs
  qh =. u8 (32 + i.16) {"1 rows                        NB. (nb,16) qh
  sc16 =. (nb,4) $ 0 (3!:4) , (48 + i.8) {"1 rows      NB. uint16 (nb,4) scales
  u16v =. (<. (0 { |: sc16) % 4096) 23 b. ((<. (1 { |: sc16) % 256) 17 b. 240) 23 b. ((<. (2 { |: sc16) % 16) 17 b. 3840) 23 b. ((3 { |: sc16) 17 b. 61440)
  d =. f16_table {~ 0 (3!:4) (1 (3!:4) u16v)
  qi =. > 0 { iq1m_idxmap
  qhi =. > 1 { iq1m_idxmap
  ibm =. > 2 { iq1m_idxmap
  lm =. > 3 { iq1m_idxmap
  jm =. > 4 { iq1m_idxmap
  qb =. qi {"1 qn                                       NB. (nb,256)
  qhb =. qhi {"1 qh                                       NB. (nb,256)
  sh =. 4 + 4 * (0 = 2 | lm)                              NB. (256,) shift: 8 even / 4 odd
  idx =. qb 23 b. ((qhb *"1 (2 ^ sh)) 17 b. 1792)         NB. (nb,256)
  grid =. iq1s_grid {~ idx                                NB. (nb,256,8)
  sce =. (<. ibm % 2) {"1 sc16                                NB. (nb,256)
  shb =. 6 * (1 = 2 | ibm)                                 NB. (256,)
  d1l =. d * (1 + 2 * ((<. sce %"1 (2 ^ shb)) 17 b. 7))
  d2l =. d * (1 + 2 * ((<. sce %"1 (2 ^ (shb + 3))) 17 b. 7))
  dl =. (d1l *"1 (lm < 2)) + d2l *"1 (lm >: 2)
  bit =. 8 + 120 * (1 = 2 | lm)                            NB. 0x08 or 0x80
  delta =. 0.125 - 0.25 * (0 ~: qhb 17 b."1 bit)
  , dl * ((qdiag ((<grid) , <jm)) + delta)
)

NB. ---- decode_tensor_flat: etype dispatch for load_tdata ----
NB. y = <etype; ne; <slice>> -> (ne,) float64. Unverified/absent types fall
NB. back to zeros (no real GGUF file carries them). 2..29 quant via the
NB. *_decode verbs; native 0/1/28/30 via gguf decoders; 24..27 via i*_decode.
decode_tensor_flat =: 3 : 0
  et =. > 0 { y
  ne =. > 1 { y
  s  =. > 2 { y
  select. et
  case. 0  do. f32_decode (<ne) , <s
  case. 1  do. f16_decode (<ne) , <s
  case. 28 do. f64_decode (<ne) , <s
  case. 30 do. decode_bf16 (<ne) , <s
  case. 2  do. q4_0_decode (<ne) , <s
  case. 3  do. q4_1_decode (<ne) , <s
  case. 6  do. q5_0_decode (<ne) , <s
  case. 7  do. q5_1_decode (<ne) , <s
  case. 8  do. q8_0_decode (<ne) , <s
  case. 10 do. q2_K_decode (<ne) , <s
  case. 11 do. q3_K_decode (<ne) , <s
  case. 12 do. q4_K_decode (<ne) , <s
  case. 13 do. q5_K_decode (<ne) , <s
  case. 14 do. q6_K_decode (<ne) , <s
  case. 15 do. q8_K_decode (<ne) , <s
  case. 20 do. iq4_nl_decode (<ne) , <s
  case. 17 do. iq2_xs_decode (<ne) , <s
  case. 22 do. iq2_s_decode (<ne) , <s
  case. 16 do. iq2_xxs_decode (<ne) , <s
  case. 18 do. iq3_xxs_decode (<ne) , <s
  case. 21 do. iq3_s_decode (<ne) , <s
  case. 19 do. iq1_s_decode (<ne) , <s
  case. 29 do. iq1_m_decode (<ne) , <s
  case. 23 do. iq4_xs_decode (<ne) , <s
  case. 24 do. i8_decode (<ne) , <s
  case. 25 do. i16_decode (<ne) , <s
  case. 26 do. i32_decode (<ne) , <s
  case. 27 do. i64_decode (<ne) , <s
  case. 9, 15 do. ne $ 0      NB. unverified/absent -> zeros
  case. do. ne $ 0
  end.
)
