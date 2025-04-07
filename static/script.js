document.addEventListener("DOMContentLoaded", function () {
    window.startCapture = function () {
        let label = document.getElementById("labelInput").value;
        if (!label) {
            alert("Please enter a label for the sign.");
            return;
        }
        fetch("/start_capture", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ label: label })
        })
        .then(response => response.json())
        .then(data => alert(data.status))
        .catch(error => console.error("Error:", error));
    };
});