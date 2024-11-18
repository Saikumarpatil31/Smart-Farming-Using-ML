// Function to redirect to the button page
function redirectToButtonsPage() {
  window.location.href = 'button.html';
}

// Add event listener to the redirect button
document.getElementById('redirect-button').addEventListener('click', function() {
  redirectToButtonsPage();
});

// Event listener for form submission (login)
document.getElementById('login-form').addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent the default form submission
  
  var username = document.getElementById('username').value;
  var password = document.getElementById('password').value;
  
  // You can add your authentication logic here
  // For this example, let's assume the username is "Praveen" and password is "Praveen@123"
  if (username === 'Praveen' && password === 'Praveen@123') {
    redirectToButtonsPage();
  } else {
    alert('Invalid username or password');
  }
});
