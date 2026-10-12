# Monthly hourly-employee demand for the past 3 years (sample data)
demand <- c(120,115,130,140,150,165,170,160,150,145,155,180,
            125,120,135,148,158,172,178,168,157,150,162,188,
            130,126,141,155,165,180,186,175,164,158,170,198)

ts_data <- ts(demand, start = c(2023, 1), frequency = 12)
plot(ts_data, main = "Monthly Hourly Employee Demand",
     ylab = "Employees", xlab = "Year")

# Holt-Winters model (captures trend and seasonality)
fit <- HoltWinters(ts_data)

# Forecast the next month
next_month <- predict(fit, n.ahead = 1)
print(next_month)

# Optional: forecast the next 3 months and plot
fc <- predict(fit, n.ahead = 3)
plot(fit, fc)
