% Pointing Loss Calculation for Horn to Parabolic Antenna using Gaussian Approximation
% Variables you can change
frequency = 10e9;            % Frequency of the transmitted signal (Hz)
D = 10;                       % Diameter of the parabolic antenna (m)
theta_error = 0.02;          % Pointing accuracy of horn antenna (degrees)

% Constants
c = 3e8;                     % Speed of light (m/s)
k = 2.77;                    % Gaussian constant for beam shape

% Calculations
lambda = c / frequency;      % Wavelength (m)
theta_HPBW = 70 * (lambda / D);  % Half-power beamwidth of the parabolic antenna (degrees)

% Pointing Loss Calculation using Gaussian Approximation
L_pointing = 10 * log10(exp(-k * (theta_error / theta_HPBW)^2));

% Convert dB loss to linear scale (power ratio)
power_ratio = 10^(L_pointing / 10);

% Calculate percentage loss
percentage_loss = (1 - power_ratio) * 100;

% Display Results
fprintf('Frequency: %.2f GHz\n', frequency / 1e9);
fprintf('Parabolic Dish Diameter: %.2f meters\n', D);
fprintf('Wavelength: %.10f meters\n', lambda);
fprintf('Half-Power Beamwidth (HPBW): %.10f degrees\n', theta_HPBW);
fprintf('Pointing Error: %.10f degrees\n', theta_error);
fprintf('Pointing Loss (Gaussian Approximation): %.10f dB\n', L_pointing);
fprintf('Percentage Loss: %.10f%%\n', percentage_loss);
