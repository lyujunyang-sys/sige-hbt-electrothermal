* IHP SG13G2 npn13G2 -- forward Gummel test (real silicon-calibrated VBIC card)
* nominal (typical) corner: all statistical multipliers = 1.0
.param vbic_cje=1 vbic_cjc=1 vbic_cjcp=1 vbic_is=1 vbic_ibei=1
.param vbic_re=1 vbic_rcx=1 vbic_rbx=1 vbic_tf=1
.param sgp_mpa_cje=1 sgp_mpa_cjc=1 sgp_mpa_is=1 sgp_mpa_bf=1
.param sgp_mpa_re=1 sgp_mpa_rb=1 sgp_mpa_rc=1
.include sg13g2_hbt_mod.lib

* subckt nodes: c b e bn(substrate)
X1 c b e 0 npn13G2 Nx=1

Vc c 0 dc 1.0
Vb b 0 dc 0.7
Ve e 0 dc 0

.control
option temp=27
dc Vb 0.3 1.04 0.02
let ic = abs(i(Vc))
let ib = abs(i(Vb))
let beta = ic/ib
wrdata gummel_27.txt ic ib beta
.endc
.end
