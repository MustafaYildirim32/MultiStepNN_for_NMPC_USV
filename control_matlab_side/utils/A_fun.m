function A_sym = A_fun(in1,in2)
%A_fun
%    A_sym = A_fun(IN1,IN2)

%    This function was generated by the Symbolic Math Toolbox version 24.1.
%    18-Jan-2025 11:50:36

x1 = in1(1,:);
x2 = in1(2,:);
x3 = in1(3,:);
t2 = abs(x2);
t3 = abs(x3);
t4 = sign(x2);
t5 = sign(x3);
mt1 = [abs(x1).*(-8.33071e-3)-x1.*sign(x1).*8.33071e-3+9.44755e-1,x2.*(-1.77022e-3)-x3.*1.06866e-1];
mt2 = [x2.*7.7453e-4+x3.*6.00072e-3,x3.*8.69158e-2,t2.*(-1.19536e-2)-x1.*1.77022e-3-t4.*x2.*1.19536e-2+9.42707e-1];
mt3 = [t2.*1.12123e-3+x1.*7.7453e-4+t4.*x2.*1.12123e-3+2.8676e-3];
mt4 = [x2.*8.69158e-2-x3.*1.306504e-3];
mt5 = [t3.*8.96787e-3-x1.*1.06866e-1+t5.*x3.*8.96787e-3-7.26335e-3];
mt6 = [t3.*(-4.35785e-2)+x1.*6.00072e-3-t5.*x3.*4.35785e-2+9.50141e-1];
A_sym = reshape([mt1,mt2,mt3,mt4,mt5,mt6],3,3);
end
