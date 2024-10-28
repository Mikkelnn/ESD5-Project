% Define the symbolic variable
syms x

% Choose the degree n for the Legendre polynomial
n = 150;  % Change this to any degree you want, e.g., n = 15

% Compute the Legendre polynomial P_n(x) using Rodrigues' formula
P_n = (1 / (2^n * factorial(n))) * diff((x^2 - 1)^n, x, n);

% Define the range of x values for plotting
x_values = linspace(-1, 1, 100);

% Substitute x_values into P_n to evaluate it numerically
y_values = double(subs(P_n, x, x_values));

% Plot the Legendre polynomial
figure;
plot(x_values, y_values, 'LineWidth', 2);
xlabel('x');
ylabel(sprintf('P_%d(x)', n));
title(sprintf('Legendre Polynomial P_%d(x) Using Rodrigues'' Formula', n));
grid on;

