from rmq.quantization import optimal_quantizer, lloyd_quantizer, distortion

for n in [3, 4, 5, 10]:
    x_newton = optimal_quantizer(n)
    x_lloyd = lloyd_quantizer(n)
    print(f"N={n}  Newton D={distortion(x_newton):.8f}  Lloyd D={distortion(x_lloyd):.8f}")
