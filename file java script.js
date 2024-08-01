const noButton = document.getElementById("no-btn");
const yesButton = document.getElementById("yes-btn");
const popup = document.getElementById("popup");

// Move the "no" button to a new random position when clicked
noButton.addEventListener("click", () => {
    noButton.style.position = "absolute";
    noButton.style.left = Math.random() * 80 + "vw";
    noButton.style.top = Math.random() * 80 + "vh";
});

// Show the popup when the "yes" button is clicked
yesButton.addEventListener("click", () => {
    popup.style.display = "block";
});

// Hide the popup when it's clicked
popup.addEventListener("click", () => {
    popup.style.display = "none";
});

// Hide the popup initially
popup.style.display = "none";


