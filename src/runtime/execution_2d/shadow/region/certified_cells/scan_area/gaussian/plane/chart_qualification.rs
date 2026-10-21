//! Independent parent-frame half-plane references at every 4x4 phase.
use super::*;
use crate::composition_2d::{Render2dBrush, Render2dColorRgba8, Render2dOpacity};
use crate::runtime::execution_2d::shadow::region::NeutralRegionBuilder;

// 100-digit Decimal evaluation of G(z)=sum_n (-1)^n*z^(2n+1)/
// (2^n*n!*(2n+1)), 180 terms. For |z|<=3 the remaining alternating
// tail is below 1e-200. F(z)=1/2+G(z)/(2G(3)); saturate outside +/-3.
// The source x>=16 is a true half-plane throughout each finite kernel
// window below; its other rectangle edges are farther than 3 sigma.
const SHEAR: [[f64; 4]; 4] = [
    [0.7741126960006841, 0.998365722580833, 1.0, 1.0],
    [0.5989735340404932, 0.9890959867645787, 1.0, 1.0],
    [0.4010264659595068, 0.9611859511515327, 1.0, 1.0],
    [0.22588730399931592, 0.8954177736818741, 1.0, 1.0],
];
const SCALED: [[f64; 4]; 4] = [
    [
        0.5989735340404932,
        0.8954177736818741,
        0.9890959867645787,
        1.0,
    ],
    [
        0.4010264659595068,
        0.7741126960006841,
        0.9611859511515327,
        0.998365722580833,
    ],
    [
        0.22588730399931592,
        0.5989735340404932,
        0.8954177736818741,
        0.9890959867645787,
    ],
    [
        0.10458222631812583,
        0.4010264659595068,
        0.7741126960006841,
        0.9611859511515327,
    ],
];

#[derive(Clone, Copy, Debug)]
enum Case {
    Shear,
    Scaled,
    Reflected,
}

impl Case {
    fn affine(self) -> Render2dAffineTransform {
        let raw = match self {
            Self::Shear => [1.0, 0.0, 0.25, 1.0, 0.0, 0.0],
            Self::Scaled => [2.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            Self::Reflected => [-2.0, 0.0, 0.5, 1.0, 64.0, 0.0],
        };
        Render2dAffineTransform::new(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]).unwrap()
    }
    fn pixel(self) -> [usize; 2] {
        [if matches!(self, Self::Shear) { 20 } else { 40 }, 16]
    }
    fn truth(self, x: usize, y: usize) -> f64 {
        match self {
            Self::Shear => SHEAR[y][x],
            Self::Scaled => SCALED[y][x],
            Self::Reflected => 1.0 - SCALED[y][x],
        }
    }

    fn alpha_byte(self) -> u8 {
        match self {
            Self::Shear => 221,
            Self::Scaled => 184,
            Self::Reflected => 71,
        }
    }
}

#[test]
fn independent_alpha_rounding_margin_exceeds_physical_error_bound() {
    // Independent alpha*255 values are 220.6397928572249,
    // 184.37233233527385 and 70.62766766472613. Their nearest UNORM8
    // rounding-boundary margins all exceed 0.12766 code values.
    // RGBA16F rounding on [0,1] contributes <=2^-12; the numerical
    // certificate contributes <=2^-20. Sixteen nonnegative f32 sample sums
    // and power-of-two normalization fit a conservative 2^-18 allowance
    // (gamma_15 < 2^-20 for the sum's relative error). Their combined
    // alpha error is <0.064 code values, so the exact byte is determined.
    let error_codes = 255.0 * (2.0_f64.powi(-12) + WIDTH / 2.0 + 2.0_f64.powi(-18));
    for case in [Case::Shear, Case::Scaled, Case::Reflected] {
        let alpha = (0..4)
            .flat_map(|y| (0..4).map(move |x| case.truth(x, y)))
            .sum::<f64>()
            / 16.0;
        let code = alpha * 255.0;
        assert_eq!(code.round() as u8, case.alpha_byte());
        let margin = 0.5 - (code - f64::from(case.alpha_byte())).abs();
        assert!(margin > 0.12766);
        assert!(error_codes < margin);
    }
}

fn request(root: [u32; 4], parent_to_root: Render2dAffineTransform) -> GaussianPlaneRequest {
    GaussianPlaneRequest {
        root,
        scale: 4.0,
        sigma: 0.125,
        offset: [0.0; 2],
        parent_to_root,
    }
}

fn source_item() -> Render2dItem {
    Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(16.0, 4.0, 16.0, 48.0).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    )
}

fn prepare(case: Case, root: [u32; 4], limit: usize) -> (PreparedPlane, usize) {
    let item = source_item();
    let mut graph = NeutralRegionBuilder::new(
        shadow::NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        },
        MAX_BYTES as u64,
    );
    let source = graph.register_source(&item, &[0, 0]).unwrap().unwrap();
    let mut fuel = limit.checked_sub(graph.graph_work).unwrap();
    let plane = source
        .prepare_gaussian_plane(request(root, case.affine()), &mut fuel, MAX_BYTES)
        .unwrap();
    plane.admit_proof_physical(&mut fuel).unwrap();
    println!(
        "PARENT_CHART {case:?} root={root:?} total_work={} remaining={fuel} retained={} visits={}",
        limit - fuel,
        plane.retained_bytes().unwrap(),
        plane.query_node_visits()
    );
    (plane, fuel)
}

#[test]
fn chart_work_host_leases_and_unqualified_affines_fail_closed() {
    let item = source_item();
    let mut graph = NeutralRegionBuilder::new(
        shadow::NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        },
        MAX_BYTES as u64,
    );
    let source = graph.register_source(&item, &[0, 0]).unwrap().unwrap();
    let root = [0, 0, 256, 256];
    let lease =
        MAX_AREA_QUERY_FIXED_BYTES as usize + 256 * 256 * size_of::<u32>() + 256 * size_of::<f64>();
    let mut fuel = 16_777_216 - graph.graph_work;
    let before = fuel;
    let plane = source
        .prepare_gaussian_plane(request(root, Case::Shear.affine()), &mut fuel, lease)
        .unwrap();
    drop(plane);
    let required = before - fuel;
    let mut exact = required;
    source
        .prepare_gaussian_plane(request(root, Case::Shear.affine()), &mut exact, lease)
        .unwrap();
    assert_eq!(
        exact, 0,
        "the same complete preparation consumes its exact receipt"
    );
    let mut short = required - 1;
    assert!(matches!(
        source.prepare_gaussian_plane(
            GaussianPlaneRequest {
                root,
                scale: 4.0,
                sigma: 0.125,
                offset: [0.0; 2],
                parent_to_root: Case::Shear.affine()
            },
            &mut short,
            lease
        ),
        Err(AreaQueryError::Resource)
    ));
    assert_eq!(short, 0);
    let before = fuel;
    assert!(matches!(
        source.prepare_gaussian_plane(
            GaussianPlaneRequest {
                root,
                scale: 4.0,
                sigma: 0.125,
                offset: [0.0; 2],
                parent_to_root: Case::Shear.affine()
            },
            &mut fuel,
            lease - 1
        ),
        Err(AreaQueryError::Resource)
    ));
    assert_eq!(
        before - fuel,
        128 + 1024 + 512,
        "reject before any numerical allocation"
    );
    for raw in [
        [1.0, 0.25, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ] {
        let affine =
            Render2dAffineTransform::new(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]).unwrap();
        assert!(matches!(
            source.prepare_gaussian_plane(
                GaussianPlaneRequest {
                    root,
                    scale: 4.0,
                    sigma: 0.125,
                    offset: [0.0; 2],
                    parent_to_root: affine
                },
                &mut fuel,
                MAX_BYTES
            ),
            Err(AreaQueryError::Unsupported)
        ));
    }
    let unrepresentable_inverse =
        Render2dAffineTransform::new(3.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap();
    assert!(matches!(
        source.prepare_gaussian_plane(
            GaussianPlaneRequest {
                root,
                scale: 4.0,
                sigma: 0.125,
                offset: [0.0; 2],
                parent_to_root: unrepresentable_inverse
            },
            &mut fuel,
            MAX_BYTES
        ),
        Err(AreaQueryError::Precision)
    ));
}

#[test]
fn complete_consuming_parent_shear_scale_reflection_stay_within_shared_work() {
    for case in [Case::Shear, Case::Scaled, Case::Reflected] {
        let (plane, remaining) = prepare(case, [0, 0, 256, 256], 16_777_216);
        assert!(remaining > 0);
        assert_eq!(plane.samples().len(), 65_536);
        assert!(plane.retained_bytes().unwrap() < MAX_BYTES);
        assert!(
            plane
                .certificates
                .errors
                .iter()
                .all(|radius| radius * 2.0 <= WIDTH)
        );
    }
}

#[test]
fn parent_chart_references_use_global_fractional_phase_and_independent_kernel_values() {
    let mut fuel = 16_777_216;
    let kernel = GaussianKernel::compiled(&mut fuel, MAX_BYTES).unwrap();
    for case in [Case::Shear, Case::Scaled, Case::Reflected] {
        let [px, py] = case.pixel();
        let chart = ParentChart::new(case.affine(), &mut fuel).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let rx = (4 * px + x) as f64 / 4.0 + 0.125;
                let ry = (4 * py + y) as f64 / 4.0 + 0.125;
                let [a, _, c, _, tx, _] = chart.inverse;
                let cx = exact_add(
                    exact_mul(a, rx).unwrap(),
                    exact_add(exact_mul(c, ry).unwrap(), tx).unwrap(),
                )
                .unwrap();
                let z = (cx - 16.0) / 0.125;
                let value = kernel.cdf(iv(z), &mut fuel).unwrap();
                let truth = case.truth(x, y);
                assert!(
                    value.lo - 2e-15 <= truth && truth <= value.hi + 2e-15,
                    "{case:?} phase{x},{y}"
                );
            }
        }
        // A crop beginning between physical pixel boundaries keeps the same
        // global centers, rather than restarting a lattice at its local zero.
        let root = [
            (4 * px - 3) as u32,
            (4 * py - 3) as u32,
            (4 * px + 5) as u32,
            (4 * py + 5) as u32,
        ];
        let (plane, _) = prepare(case, root, 16_777_216);
        assert_eq!(plane.root(), root);
        assert_eq!(plane.extent(), [8, 8]);
    }
}

fn assert_phase_references(plane: &PreparedPlane, values: &[f32], case: Case) -> f64 {
    let [px, py] = case.pixel();
    let [x0, y0, _, _] = plane.root();
    let width = plane.extent()[0] as usize;
    let mut alpha = 0.0;
    for y in 0..4 {
        for x in 0..4 {
            let word =
                plane.samples[((py * 4 + y) - y0 as usize) * width + px * 4 + x - x0 as usize];
            let (value, radius) = if word & QUERY != 0 {
                let index = (word & !QUERY) as usize;
                (f64::from(values[index]), plane.certificates.errors[index])
            } else {
                (f64::from(word) / Q24, plane.certificates.literal_radius)
            };
            let truth = case.truth(x, y);
            assert!(
                (value - truth).abs() <= radius,
                "{case:?} phase{x},{y}: {value} +/- {radius}, expected{truth}"
            );
            alpha += truth / 16.0;
        }
    }
    alpha
}

#[test]
#[ignore = "requires coordinated Vulkan: consuming-parent chart sixteen phase references"]
fn consuming_parent_chart_matches_independent_phases_on_vulkan() {
    for case in [Case::Shear, Case::Scaled, Case::Reflected] {
        let (plane, _) = prepare(case, [0, 0, 256, 256], 16_777_216);
        let values = plane.queries.execute_proof();
        let pixels =
            crate::runtime::execution_2d::lowering::sample_space::gaussian_gpu_proof::execute(
                plane.samples(),
                plane.extent(),
                plane.query_data().0,
                plane.query_data().1,
            );
        let [px, py] = case.pixel();
        let alpha = assert_phase_references(&plane, &values, case);
        let red = if alpha <= 0.0031308 {
            12.92 * alpha
        } else {
            1.055 * alpha.powf(1.0 / 2.4) - 0.055
        };
        let expected_red = (255.0 * red).round() as u8;
        let offset = (py * 64 + px) * 4;
        let actual = &pixels[offset..offset + 4];
        assert!(
            actual[0].abs_diff(expected_red) <= 2,
            "{case:?} red: {actual:?}, expected {expected_red}"
        );
        assert_eq!(&actual[1..3], &[0, 0], "{case:?} green/blue");
        assert_eq!(actual[3], case.alpha_byte(), "{case:?} alpha");
        drop((plane, values, pixels));
        let root = [
            (4 * px - 3) as u32,
            (4 * py - 3) as u32,
            (4 * px + 5) as u32,
            (4 * py + 5) as u32,
        ];
        let (crop, _) = prepare(case, root, 16_777_216);
        let values = crop.queries.execute_proof();
        // Query readback qualifies the global phase of a non-pixel-aligned
        // crop. Final compositor pixels above use an aligned complete plane.
        assert_phase_references(&crop, &values, case);
    }
}
