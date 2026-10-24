// Arithmetic conformance ABI. Q24 is checked for each completed test case;
// this is not a coverage-node conversion or an image-composition entry point.
struct Input {
    opcode: u32,
    padding: u32,
    a: vec2<u32>,
    b: vec2<u32>,
    coefficients: array<vec2<u32>, 13>,
};
@group(0) @binding(0) var<storage, read> inputs: array<Input>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<u32>>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index=invocation.x;
    if index>=arrayLength(&inputs) { return; }
    let input=inputs[index];
    var result=vec2<u32>(0u);
    if input.opcode==0u {
        result=multiply(input.a,input.b);
    } else if input.opcode==4u {
        result=input.a;
        for (var i=0u; i<13u; i=i+1u) { result=add(result,input.coefficients[i]); }
    } else {
        result=input.coefficients[12];
        for (var i=12u; i>0u; i=i-1u) {
            result=add(multiply(result,input.a),input.coefficients[i-1u]);
        }
        if input.opcode==2u { result=add(vec2<u32>(0u,32768u),result); }
        if input.opcode==3u { result=add(vec2<u32>(0u,32768u),negate(result)); }
    }
    outputs[index]=vec4<u32>(result,completed_q24_bits(result),0u);
}
