function [diff_angle,psi_val_out,target_out] = Angle_Diff(psi_value,target)
    if abs(psi_value) > pi
        psi_value = mod(psi_value + pi, 2*pi) - pi;
    end
    if abs(target) > pi
        target = mod(target + pi, 2*pi) - pi;
    end
    diff_angle = target-psi_value;
    if abs(diff_angle) > pi
        diff_angle = mod(diff_angle + pi, 2*pi) - pi;
    end
    psi_val_out = psi_value;
    target_out = target;
   
end
