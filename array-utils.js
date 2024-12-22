function argSort(arr) {
    return arr
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value)
        .map((item) => item.index);
}

module.exports = {
    argSort
};