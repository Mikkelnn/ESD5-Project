% Define constants
Kp = 20; % Example value for Kp
Ki = 0;  % Example value for Ki
Kd = 10; % Example value for Kd

% Define numerator and denominator coefficients
numerator = [Kd, Kp, Ki];
denominator = [2, Kd, Kp, Ki];

% Create transfer function
sys = tf(numerator, denominator)

% Plot Step Response
figure;
step(sys);
grid on;
title('Step Response of the System');

% Get Step Response Information
step_info = stepinfo(sys);

% Display Rise Time
disp('Step Response Information:');
disp(step_info);

% Extract and Display Rise Time
rise_time = step_info.RiseTime;
disp(['Rise Time: ', num2str(rise_time), ' seconds']);