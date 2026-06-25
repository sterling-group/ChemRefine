// Make the header site title clickable (navigates home, like the logo does).
// Material for MkDocs renders two .md-header__topic elements:
//   [0] = site name (always visible at the top, hidden on scroll)
//   [1] = page/section title (shown on scroll)
// Only the site name (index 0) should link home.
document.addEventListener("DOMContentLoaded", function () {
  var topics = document.querySelectorAll(".md-header__topic");
  var logo = document.querySelector(".md-header__button.md-logo");
  if (topics.length > 0 && logo) {
    topics[0].style.cursor = "pointer";
    topics[0].addEventListener("click", function () {
      window.location.href = logo.href;
    });
  }
});
