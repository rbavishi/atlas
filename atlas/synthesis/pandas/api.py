from atlas import generator


@generator
def pivot_outputs(inp):
    col = Select(inp.columns)
    idx = Select(inp.columns)
    val = Select(inp.columns)

    try:
        return inp.pivot(index=idx, columns=col, values=val)

    except:
        return None