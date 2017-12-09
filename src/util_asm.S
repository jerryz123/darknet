        .text
        .align 3
        .globl vgather
vgather:
        vpset   vp0
        vlw     vv0, va1 #INDEXES
        vadd    vv1, vs0, vs0
        vcmplt  vp1, vv0, vs0
@!vp1   vlxw    vv1, vs1, vv0

        vsw     vv1, va2
        vstop

        .globl vgather_h        
vgather_h:
        vpset   vp0
        vlw     vv0, va1 #INDEXES
        vadd    vv1, vs0, vs0
        vcmplt  vp1, vv0, vs0
@!vp1   vlxh    vv1, vs1, vv0

        vsh     vv1, va2
        vstop

        .globl  vacc_gather_pre
vacc_gather_pre:
        vpset   vp0
        vlw     vv0, va0 #INDEXES
        vadd    vv2, vs0, vs0
        vstop

        .globl  vacc_gather
vacc_gather:
        vlxw    vv1, vs1, vv0
        vfadd.s vv2, vv2, vv1
        vstop

        .globl  vacc_gather_post
vacc_gather_post:
        vsw     vv2, va0
        vstop



	.globl sgemm_opt_v_4_4_vpset
	.globl sgemm_opt_v_4_4
	.globl sgemm_opt_v_4_4_edge0
	.globl sgemm_opt_v_4_4_pre
	.globl sgemm_opt_v_4_4_post
	.globl sgemm_opt_v_4_4_edge1
	.globl sgemm_opt_v_4_4_pre_edge
	.globl sgemm_opt_v_4_4_post_edge

	.align 3
sgemm_opt_v_4_4_vpset:
	vpset vp0
	vstop

	.align 3
sgemm_opt_v_4_4_pre:
	vlw vv0, va0 # c0
	vlw vv1, va1 # c1
	vlw vv2, va2 # c2
	vlw vv3, va3 # c3
	vstop


	.align 3
sgemm_opt_v_4_4_pre_edge:
	vlw vv0, va0 # c0
	vstop

	.align 3
sgemm_opt_v_4_4:
	vlw vv4, va4                    # b0
	vfmadd.s vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.s vv1, vv4, vs5, vv1     # c1 += a10 * b0

	vlw vv5, va5                    # b1
	vfmadd.s vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.s vv3, vv4, vs13, vv3    # c3 += a30 * b0

	vlw vv6, va6                    # b2
	vfmadd.s vv0, vv5, vs2, vv0     # c0 += a01 * b1
	vfmadd.s vv1, vv5, vs6, vv1     # c1 += a11 * b1

	vlw vv7, va7                    # b3
	vfmadd.s vv0, vv6, vs3, vv0     # c0 += a02 * b2
	vfmadd.s vv1, vv6, vs7, vv1     # c1 += a12 * b2
	vfmadd.s vv0, vv7, vs4, vv0     # c0 += a03 * b3
	vfmadd.s vv1, vv7, vs8, vv1     # c1 += a13 * b3
	vfmadd.s vv2, vv5, vs10, vv2    # c2 += a21 * b1
	vfmadd.s vv3, vv5, vs14, vv3    # c3 += a31 * b1
	vfmadd.s vv2, vv6, vs11, vv2    # c2 += a22 * b2
	vfmadd.s vv3, vv6, vs15, vv3    # c3 += a32 * b2
	vfmadd.s vv2, vv7, vs12, vv2    # c2 += a23 * b3
	vfmadd.s vv3, vv7, vs16, vv3    # c3 += a33 * b3
	vstop
	.align 3
sgemm_opt_v_4_4_edge0:
	vlw vv4, va4                    # b0
	vfmadd.s vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.s vv1, vv4, vs5, vv1     # c1 += a10 * b0
	vfmadd.s vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.s vv3, vv4, vs13, vv3    # c3 += a30 * b0
	vstop
	.align 3
sgemm_opt_v_4_4_edge1:
	vlw vv4, va4                    # b0
	vfmadd.s vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vstop
	.align 3
sgemm_opt_v_4_4_post:
        vsw vv0, va0 # c0
        vsw vv1, va1 # c1
        vsw vv2, va2 # c2
        vsw vv3, va3 # c3
        vstop
        .align 3
sgemm_opt_v_4_4_post_edge:
        vsw vv0, va0 # c0
        vstop


        .globl vfill_rv_cpu
        .align 3
vfill_rv_cpu:
        vpset vp0
        vadd vv0, vs0, vs1
        vsstw vv0, va0, va1
        vstop

        .globl vfill_rv_cpu_1
        .align 3
vfill_rv_cpu_1:
        vpset vp0
        vadd vv0, vs0, vs1
        vsw vv0, va0
        vstop

        .globl vfill_rv_cpu_h
        .align 3
vfill_rv_cpu_h:
        vpset vp0
        vfcvt.h.s vv0, vs1
        vssth vv0, va0, va1
        vstop

        .globl vfill_rv_cpu_h_1
        .align 3
vfill_rv_cpu_h_1:
        vpset vp0
        vfcvt.h.s vv0, vs1
        vsh vv0, va0
        vstop

        .globl vcvt_sh
        .align 3
vcvt_sh:
        vpset vp0
        vlw vv0, va0
        vfcvt.h.s vv1, vv0
        vsh vv1, va1
        vstop

        .globl vcvt_hs
        .align 3
vcvt_hs:
        vpset vp0
        vlh vv1, va0
        vfcvt.s.h vv0, vv1
        vsw vv0, va1
        vstop


        .globl vscalex
        .align 3
vscalex:
        vpset vp0
        vlw vv0, va0
        vfmul.s vv0, vv0, vs1
        vsw vv0, va0
        vstop
        .globl vscalex_h
        .align 3
vscalex_h:
        vpset vp0
        vlh vv0, va0
        vfcvt.h.s vs1, vs1
        vfmul.h vv0, vv0, vs1
        vsh vv0, va0
        vstop



	.globl hgemm_opt_v_4_4_vpset
	.globl hgemm_opt_v_4_4
	.globl hgemm_opt_v_4_4_edge0
	.globl hgemm_opt_v_4_4_pre
	.globl hgemm_opt_v_4_4_post
	.globl hgemm_opt_v_4_4_edge1
	.globl hgemm_opt_v_4_4_pre_edge
	.globl hgemm_opt_v_4_4_post_edge

	.align 3
hgemm_opt_v_4_4_vpset:
	vpset vp0
	vstop

	.align 3
hgemm_opt_v_4_4_pre:
	vlh vv0, va0 # c0
	vlh vv1, va1 # c1
	vlh vv2, va2 # c2
	vlh vv3, va3 # c3
	vstop
	.align 3
hgemm_opt_v_4_4_pre_edge:
	vlh vv0, va0 # c0
	vstop

	.align 3
hgemm_opt_v_4_4:
	vlh vv4, va4                    # b0
	vfmadd.h vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.h vv1, vv4, vs5, vv1     # c1 += a10 * b0

	vlh vv5, va5                    # b1
	vfmadd.h vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.h vv3, vv4, vs13, vv3    # c3 += a30 * b0

	vlh vv6, va6                    # b2
	vfmadd.h vv0, vv5, vs2, vv0     # c0 += a01 * b1
	vfmadd.h vv1, vv5, vs6, vv1     # c1 += a11 * b1

	vlh vv7, va7                    # b3
	vfmadd.h vv0, vv6, vs3, vv0     # c0 += a02 * b2
	vfmadd.h vv1, vv6, vs7, vv1     # c1 += a12 * b2
	vfmadd.h vv0, vv7, vs4, vv0     # c0 += a03 * b3
	vfmadd.h vv1, vv7, vs8, vv1     # c1 += a13 * b3
	vfmadd.h vv2, vv5, vs10, vv2    # c2 += a21 * b1
	vfmadd.h vv3, vv5, vs14, vv3    # c3 += a31 * b1
	vfmadd.h vv2, vv6, vs11, vv2    # c2 += a22 * b2
	vfmadd.h vv3, vv6, vs15, vv3    # c3 += a32 * b2
	vfmadd.h vv2, vv7, vs12, vv2    # c2 += a23 * b3
	vfmadd.h vv3, vv7, vs16, vv3    # c3 += a33 * b3
	vstop
	.align 3
hgemm_opt_v_4_4_edge0:
	vlh vv4, va4                    # b0
	vfmadd.h vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vfmadd.h vv1, vv4, vs5, vv1     # c1 += a10 * b0
	vfmadd.h vv2, vv4, vs9, vv2     # c2 += a20 * b0
	vfmadd.h vv3, vv4, vs13, vv3    # c3 += a30 * b0
	vstop
	.align 3
hgemm_opt_v_4_4_edge1:
	vlh vv4, va4                    # b0
	vfmadd.h vv0, vv4, vs1, vv0     # c0 += a00 * b0
	vstop
	.align 3
hgemm_opt_v_4_4_post:
        vsh vv0, va0 # c0
        vsh vv1, va1 # c1
        vsh vv2, va2 # c2
        vsh vv3, va3 # c3
        vstop
        .align 3
hgemm_opt_v_4_4_post_edge:
        vsh vv0, va0 # c0
        vstop

        .globl vcopy_oo
        .align 3
vcopy_oo:
        vpset vp0
        vlw vv0, va0
        vsw vv0, va1
        vstop


        .globl vcopy_oo_h
        .align 3
vcopy_oo_h:
        vpset vp0
        vlh vv0, va0
        vsh vv0, va1
        vstop

        .globl vnormalize_cpu
        .align 3
vnormalize_cpu:
        vpset vp0
        vlw vv0, va0
        vfsqrt.s vs4, vs2
        vfadd.s vs4, vs4, vs3
        vfsub.s vv0, vv0, vs1
        vfdiv.s vv0, vv0, vs4
        vsw vv0, va0
        vstop


        .globl vnormalize_cpu_h
        .align 3
vnormalize_cpu_h:
        vpset vp0
        vlh vv0, va0
        vfcvt.h.s vs5, vs3
        vfsqrt.h vs4, vs2
        vfadd.h vs4, vs4, vs5
        vfsub.h vv0, vv0, vs1
        vfdiv.h vv0, vv0, vs4
        vsh vv0, va0
        vstop

        .globl vaddx
        .align 3
vaddx:
        vpset vp0
        vlw vv0, va0
        vfadd.s vv0, vv0, vs1
        vsw vv0, va0
        vstop


        .globl vaddx_h
        .align 3
vaddx_h:
        vpset vp0
        vlh vv0, va0
        vfadd.h vv0, vv0, vs1
        vsh vv0, va0
        vstop

        .globl vleaky_activate
        .align 3
vleaky_activate:
        vpset vp0
        vlw vv0, va0
        vfcvt.s.d vs1, vs0
        vcmpfle.s vp1, vv0, vs1
        @vp1 vfmul.s vv0, vv0, vs2
        @vp1    vsw vv0, va0
        vstop


        .globl vleaky_activate_h
        .align 3
vleaky_activate_h:
        vpset vp0
        vlh vv0, va0
        vfcvt.h.s vs3, vs2
        vfcvt.h.d vs1, vs0
        vcmpfle.h vp1, vv0, vs1
        @vp1 vfmul.h vv0, vv0, vs3
        @vp1    vsh vv0, va0
        vstop

        .globl vhwacha_memcpy
        .align 3
vhwacha_memcpy:
        vpset vp0
        vlh vv0, va0
        vsh vv0, va0
        vstop

        .globl hgemm_eff_vpset
        .globl hgemm_eff_pre
        .globl hgemm_eff_top
        .globl hgemm_eff_0
        .globl hgemm_eff_1
        .globl hgemm_eff_2
        .globl hgemm_eff_3
        .globl hgemm_eff_4
        .globl hgemm_eff_5
        .globl hgemm_eff_6
        .globl hgemm_eff_7
        .globl hgemm_eff_post
        .globl hgemm_eff_pre_edge
        .globl hgemm_eff_post_edge
        .align 3
hgemm_eff_vpset:
        vpset vp0
        vstop
hgemm_eff_pre:
        vlh vv0, va0
        vlh vv1, va1
        vlh vv2, va2
        vlh vv3, va3
        vlh vv4, va4
        vlh vv5, va5
        vlh vv6, va6
        vlh vv7, va7
        vstop
hgemm_eff_top:
        vlh vv8, va8
        vstop
hgemm_eff_0:
        vfmadd.h vv0, vv8, vs1, vv0
        vstop
hgemm_eff_1:
        vfmadd.h vv1, vv8, vs2, vv1
        vstop
hgemm_eff_2:
        vfmadd.h vv2, vv8, vs3, vv2
        vstop
hgemm_eff_3:
        vfmadd.h vv3, vv8, vs4, vv3
        vstop
hgemm_eff_4:
        vfmadd.h vv4, vv8, vs5, vv4
        vstop
hgemm_eff_5:
        vfmadd.h vv5, vv8, vs6, vv5
        vstop
hgemm_eff_6:
        vfmadd.h vv6, vv8, vs7, vv6
        vstop
hgemm_eff_7:
        vfmadd.h vv7, vv8, vs8, vv7
        vstop
hgemm_eff_post:
        vsh vv0, va0
        vsh vv1, va1
        vsh vv2, va2
        vsh vv3, va3
        vsh vv4, va4
        vsh vv5, va5
        vsh vv6, va6
        vsh vv7, va7
        vstop
hgemm_eff_pre_edge:
        vlh vv0, va0
        vstop
hgemm_eff_post_edge:
        vsh vv0, va0
