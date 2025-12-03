function argSort(arr: Array<number>) {
    return arr
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value)
        .map((item) => item.index);
}

export { argSort };