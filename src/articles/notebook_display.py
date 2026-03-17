"""Helpers for rendering Spark DataFrames in notebooks."""


def spark_df_to_scrollable_html(df, rows=5):
    preview_pdf = df.limit(rows).toPandas()
    table_html = preview_pdf.to_html(index=False, border=0)
    return (
        '<div style="max-width:100%;overflow-x:auto;white-space:nowrap;">'
        f"{table_html}"
        "</div>"
    )
