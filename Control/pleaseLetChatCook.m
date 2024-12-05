% MATLAB Script for Gain and Phase Margin Analysis

% Define the PD Controller Parameters
Kp = 20;  % Proportional gain
Ki = 0.1; % Integral gain
Kd = 10;  % Derivative gain

% Define the Plant Transfer Function (0.5/s^2)
numerator_plant = 0.5;
denominator_plant = [1,0,0]; % s^2
Plant = tf(numerator_plant, denominator_plant);

% PD Controller Transfer Function
numerator_PD = [Kd Ki Kp]; % Kd*s + Kp
denominator_PD = [1,0];     % PD controller doesn't add poles
PD_Controller = tf(numerator_PD, denominator_PD);

% Open-Loop Transfer Function
Open_Loop = PD_Controller * Plant;
Open_Loop

% Step Input (Step time = 0, Initial value = 0, Final value = 1)
step_time = 0; % No delay in step input
initial_value = 0;
final_value = 1;

% Generate the Closed-Loop System (Feedback with Unity Gain)
Closed_Loop = feedback(Open_Loop, 1);
%Closed_Loop_Numerator = PD_Controller * Plant;
%Closed_Loop_Denominator = 1 + PD_Controller * Plant * 1;
%Closed_Loop = tf(Closed_Loop_Numerator,Closed_Loop_Denominator);
%Closed_Loop

% Analyze the Bode Plot and Margins
figure;
margin(Open_Loop); % Bode plot with gain and phase margins
grid on;

% Display the Gain Margin and Phase Margin Numerically
[Gm, Pm, Wcg, Wcp] = margin(Open_Loop);
fprintf('Gain Margin (dB): %.2f\n', 20*log10(Gm));
fprintf('Phase Margin (degrees): %.2f\n', Pm);
fprintf('Gain Crossover Frequency (rad/s): %.2f\n', Wcg);
fprintf('Phase Crossover Frequency (rad/s): %.2f\n', Wcp);

% Step Response of Closed-Loop System
figure;
step(final_value * Closed_Loop);
title('Step Response of Closed-Loop System');
grid on;
