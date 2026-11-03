#let report-style(body) = {
  set page(
    paper: "us-letter",
    fill: white,
    margin: (x: 0.8in, top: 0.7in, bottom: 0.65in),
    footer: context {
      set text(size: 8pt, fill: black)
      grid(
        columns: (1fr, auto),
        [dbTwin synthetic RNA-Seq benchmark report],
        counter(page).display(),
      )
    },
  )
  set text(size: 10pt, fill: black)
  set par(leading: 0.62em, justify: true)
  body
}

#let section-heading(title, subsection: none) = {
  heading(level: 1, outlined: false)[#title]
  line(length: 100%, stroke: 0.8pt + black)
  if subsection != none {
    v(14pt)
    heading(level: 2, outlined: false)[#subsection]
  }
}

#let report-table(columns, rows, column-widths: (1.8fr, 1.1fr, 0.6fr), width: 100%, cell-align: center + horizon, justify: true, inset: 7pt) = {
  set text(size: 8.5pt)
  set par(justify: justify)
  align(center, block(width: width, table(
    columns: column-widths,
    align: cell-align,
    inset: inset,
    stroke: (x: 0.5pt + black, y: none),
    fill: white,
    table.hline(stroke: 0.5pt + black),
    table.header(
      ..columns.map(column => [*#column*]),
      table.hline(stroke: 0.5pt + black),
    ),
    ..rows.flatten().map(value => [#value]),
    table.hline(stroke: 0.5pt + black),
  )))
}

#let report-figure(path, number, caption, width: 4in) = {
  block(breakable: false, width: 100%)[
    #align(center, image(path, width: width))
    #v(4pt)
    #text(size: 8.5pt)[*Figure #number.* #caption]
  ]
}

#let two-column-figures(figures) = {
  grid(
    columns: if figures.len() == 1 { (1fr,) } else { (1fr, 1fr) },
    gutter: 18pt,
    ..figures,
  )
}
