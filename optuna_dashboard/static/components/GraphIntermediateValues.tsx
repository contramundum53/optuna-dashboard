import * as plotly from "plotly.js-dist"
import React, { FC, useEffect } from "react"
import {Grid, Typography, useTheme} from "@mui/material"
import {plotlyDarkTemplate} from "./PlotlyDarkMode";

const plotDomId = "graph-intermediate-values"

export const GraphIntermediateValues: FC<{
  trials: Trial[]
}> = ({ trials = [] }) => {
  const theme = useTheme()
  useEffect(() => {
    plotIntermediateValue(trials, theme.palette.mode)
  }, [trials, theme.palette.mode])
  return (
    <Grid container direction="row">
      <Grid item xs={3}>
        <Grid container direction="column">
          <Typography variant="h6" sx={{ margin: "1em 0" }}>
            Intermediate values
          </Typography>
        </Grid>
      </Grid>

      <Grid item xs={9}>
        <div id={plotDomId} />
      </Grid>
    </Grid>
  )
}

const plotIntermediateValue = (trials: Trial[], mode: string) => {
  if (document.getElementById(plotDomId) === null) {
    return
  }

  const layout: Partial<plotly.Layout> = {
    margin: {
      l: 50,
      t: 0,
      r: 50,
      b: 0,
    },
    template: mode === "dark" ? plotlyDarkTemplate : {},
  }
  if (trials.length === 0) {
    plotly.react(plotDomId, [], layout)
    return
  }

  const filteredTrials = trials.filter(
    (t) =>
      t.state === "Complete" ||
      (t.state === "Pruned" && t.values && t.values.length > 0)
  )
  const plotData: Partial<plotly.PlotData>[] = filteredTrials.map((trial) => {
    return {
      x: trial.intermediate_values.map((iv) => iv.step),
      y: trial.intermediate_values.map((iv) => iv.value),
      mode: "lines+markers",
      type: "scatter",
      name: `trial #${trial.number}`,
    }
  })
  plotly.react(plotDomId, plotData, layout)
}
