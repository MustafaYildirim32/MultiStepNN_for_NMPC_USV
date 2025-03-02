function out = wrapToPi(in)
    if abs(in) > pi
        in = mod(in + pi, 2*pi) - pi;
    end
    out=in;
end

