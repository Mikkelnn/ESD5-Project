% Define the continuous function h(x) = sin(8x)
f = 2;  % Frequency in radians for sin(8x)
Fs = 3.333;  % Sampling frequency (0.9 Hz)
du = 100;
Ts = 1/Fs;  % Sampling period
t = 0:Ts:du;  % Time vector (up to 10 seconds)
h = sin(2*pi*f * t);  % Sampled version of the function h(x)

% Create a fine time vector to plot the continuous function
t_cont = 0:0.001:du;  % Finer time vector for smooth plot
h_cont = sin(2*pi*f * t_cont);  % Continuous function

% Define the sampled frequency sine wave
f_sampled_wave = Ts;  % Frequency of the sampled sine wave (0.9 Hz)
sampled_sine_wave = sin( f_sampled_wave * t_cont);  % Sampled sine wave

% Plot the continuous-time function
figure;
plot(t_cont, h_cont, 'b', 'LineWidth', 1.5); % Continuous function in blue
hold on;

% Plot the sampled points
stem(t, h, 'r', 'LineWidth', 1.5);  % Sampled points in red

% Labels and title
title('Continuous and Sampled Signal with Sampled Frequency Sinusoid');
xlabel('Time (seconds)');
ylabel('Amplitude');
legend('Continuous Signal: sin(8x)', 'Sampled Points');
grid on;
