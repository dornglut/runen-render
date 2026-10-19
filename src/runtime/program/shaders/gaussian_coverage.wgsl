// Gaussian coverage: normalized coefficients, contiguous queries and signed nodes.
// CPU admission proves all offsets/ranges and the absolute reduction bound.
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4<u32>>;
fn pair(index: u32) -> vec2<u32> { return vec2<u32>(words[index], words[index+1u]); }
fn unit_clamp(value: vec2<u32>) -> vec2<u32> {
    if negative(value) { return vec2<u32>(0u); }
    if value.y >= 65536u { return vec2<u32>(0u,65536u); }
    return value;
}
fn kernel(z: vec2<u32>, kind: u32) -> vec2<u32> {
    var magnitude=z;
    if negative(z) { magnitude=negate(z); }
    if magnitude.y>=196608u {
        if kind==1u || negative(z) { return vec2<u32>(0u); }
        return vec2<u32>(0u,65536u);
    }
    // Q48 patch width is 3*2^44, divisible by 2^32. Integer high-word
    // division chooses the exact patch; no rounded floating seam test.
    let patch_index=magnitude.y/12288u;
    let center=vec2<u32>(0u,(2u*patch_index+1u)*6144u);
    let delta=add(magnitude,negate(center));
    let table=4u+(patch_index*2u+kind)*26u;
    var value=pair(table+24u);
    for (var i=12u; i>0u; i=i-1u) {
        value=add(multiply(value,delta),pair(table+2u*(i-1u)));
    }
    if kind==0u {
        if negative(z) { value=negate(value); }
        value=add(vec2<u32>(0u,32768u),value);
    }
    // True normalized CDF and density are in [0,1]. This projection cannot
    // increase their error and supplies the exact integer product bounds.
    return unit_clamp(value);
}
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let query=invocation.x;
    let count=words[0];
    if query>=count { return; }
    let header=836u+5u*query;
    let first=words[header];
    let length=words[header+1u];
    var sum=vec2<u32>(0u);
    let nodes=836u+5u*count;
    for (var i=0u; i<length; i=i+1u) {
        let node=nodes+6u*(first+i);
        let cdf=kernel(pair(node),0u);
        let density=kernel(pair(node+2u),1u);
        let weight=pair(node+4u);
        sum=add(sum,multiply(multiply(cdf,density),weight));
    }
    if words[header+4u]!=0u { sum=negate(sum); }
    sum=add(pair(header+2u),sum);
    results[query]=vec4<u32>(sum,completed_q24_bits(sum),0u);
}

// A disposable scalar field lookup. These samples are never source geometry.
// Compute and lookup use the same completed Q24 query values with no CPU readback.
@group(0) @binding(2) var<storage, read> field_samples: array<u32>;
@group(0) @binding(3) var<storage, read> field_queries: array<vec4<u32>>;
@group(0) @binding(4) var<uniform> field_extent: vec4<u32>;
@vertex fn field_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let vertices = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
    return vec4<f32>(vertices[index],0.0,1.0);
}
@fragment fn field_fragment(@builtin(position) position: vec4<f32>) -> @location(0) f32 {
    // CPU admission proves local+offset is in the signed i32 domain.
    // Crop before converting to u32, preserving negative tile/map origins.
    let offset=bitcast<vec2<i32>>(field_extent.zw);
    let sample=vec2<i32>(position.xy)+offset;
    if any(sample<vec2<i32>(0)) { return 0.0; }
    let point=vec2<u32>(sample);
    if point.x>=field_extent.x || point.y>=field_extent.y { return 0.0; }
    let value=field_samples[point.y*field_extent.x+point.x];
    if (value & 2147483648u)!=0u { return bitcast<f32>(field_queries[value & 2147483647u].z); }
    return f32(value)*(1.0/16777216.0);
}
