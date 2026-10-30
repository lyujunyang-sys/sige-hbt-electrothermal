* IHP SG13G2 npn13G2 -- generate REAL reference dataset (VBIC, silicon-calibrated)
.param vbic_cje=1 vbic_cjc=1 vbic_cjcp=1 vbic_is=1 vbic_ibei=1
.param vbic_re=1 vbic_rcx=1 vbic_rbx=1 vbic_tf=1
.param sgp_mpa_cje=1 sgp_mpa_cjc=1 sgp_mpa_is=1 sgp_mpa_bf=1
.param sgp_mpa_re=1 sgp_mpa_rb=1 sgp_mpa_rc=1
.include sg13g2_hbt_mod.lib

X1 c b e 0 npn13G2 Nx=1
Vc c 0 dc 1.0
Vb b 0 dc 0.7
Ve e 0 dc 0

.control
* --- forward Gummel at several chuck temperatures (Vce = 1.0 V) ---
foreach tt -40 27 85 125
  option temp=$tt
  alter Vc = 1.0
  dc Vb 0.3 1.00 0.01
  wrdata gummel_T$tt abs(i(Vc)) abs(i(Vb))
end

* --- output characteristics Ic-Vce at stepped Vbe (T = 27 C) ---
option temp=27
foreach vbb 0.70 0.75 0.80 0.85
  alter Vb = $vbb
  dc Vc 0.0 1.6 0.02
  wrdata output_Vbe$vbb abs(i(Vc))
end
.endc
.end
