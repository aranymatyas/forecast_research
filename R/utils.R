library(dplyr)
library(ggplot2)
library(plotly)

plot_forecast <- function(fc, test_ts, target_id = NULL,
                                     x_label = "Date", y_label = "Web Traffic") {

  # Handle the default target_id
  if (is.null(target_id)) {
    target_id <- unique(fc$unique_id)[1]
  }

  # Filter the data
  fc_single <- fc %>% filter(unique_id == target_id)
  test_single <- test_ts %>% filter(unique_id == target_id)

  # Build the ggplot object
  p <- autoplot(fc_single, level = NULL) +
    geom_line(
      data = test_single,
      aes(x = ds, y = y),
      color = "black",
      linewidth = 1
    ) +
    theme_minimal() +
    labs(
      title = paste("Forecast vs Actuals for:", target_id),
      y = y_label,
      x = x_label
    )

  return(ggplotly(p))
}

error_metrics <- function(fc, ts) {
  eval_metrics <- fc %>%
    accuracy(ts) %>% # This handles NaN in validation set
    select(unique_id, .model, MAPE, MASE)

  return(eval_metrics)
}