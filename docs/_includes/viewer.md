<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  // The structure to show is not written here: each tutorial sets `data-xyz` on its own
  // `#viewer` div, relative to the repository root. That one attribute is the only thing
  // that differed between the six copies of this block that used to exist.
  (() => {
    const el = document.getElementById("viewer");
    const viewer = $3Dmol.createViewer(el, { backgroundColor: "white" });
    const base = "https://raw.githubusercontent.com/sterling-group/ChemRefine/main/";

    fetch(base + el.dataset.xyz)
      .then(r => r.text())
      .then(data => {
        viewer.addModel(data, "xyz");   // force XYZ format
        viewer.setStyle({}, {stick:{radius:0.15}, sphere:{scale:0.25}});
        viewer.zoomTo();
        viewer.render();
      })
      .catch(err => console.error("Could not load XYZ:", err));
  })();
</script>
