// Signed Q48 Gaussian arithmetic. CPU admission proves signed ranges.
// Unsigned schoolbook product uses B=2^16:
// (B-1)^2+(B-1)+(B-1)=2^32-1, so each MAC is exact u32.
// The updated limb and carry are again in [0,B-1]. At the end of row i,
// limb i+4 has not previously been written, so storing carry needs no add.
fn negative(a: vec2<u32>) -> bool { return (a.y & 0x80000000u) != 0u; }
fn negate(a: vec2<u32>) -> vec2<u32> {
    let low = (~a.x) + 1u;
    return vec2<u32>(low, (~a.y) + select(0u, 1u, low == 0u));
}
fn add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}
fn limbs(a: vec2<u32>) -> array<u32, 4> {
    return array<u32, 4>(a.x & 65535u, a.x >> 16u, a.y & 65535u, a.y >> 16u);
}
fn multiply(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var magnitude_a = a;
    var magnitude_b = b;
    if negative(a) { magnitude_a = negate(a); }
    if negative(b) { magnitude_b = negate(b); }
    let x = limbs(magnitude_a);
    let y = limbs(magnitude_b);
    var product: array<u32, 8>;
    for (var i=0u; i<4u; i=i+1u) {
        var carry=0u;
        for (var j=0u; j<4u; j=j+1u) {
            let index=i+j;
            let total=x[i]*y[j]+product[index]+carry;
            product[index]=total & 65535u;
            carry=total >> 16u;
        }
        product[i+4u]=carry;
    }
    // Extract bits [48,112) explicitly. A u32 >> 48 would shift by 16
    // under WGSL semantics and would be incorrect. CPU proves upper bits zero.
    var result=vec2<u32>(product[3] | (product[4]<<16u), product[5] | (product[6]<<16u));
    if negative(a) != negative(b) { result=negate(result); }
    return result;
}
fn completed_q24_bits(value: vec2<u32>) -> u32 {
    if negative(value) { return 0u; }
    if value.y>=65536u { return 0x3f800000u; }
    let q24=(value.x>>24u) | (value.y<<8u);
    // q24<=2^24, exactly representable. Binary scaling is also exact.
    return bitcast<u32>(f32(q24)*0.000000059604644775390625f);
}
