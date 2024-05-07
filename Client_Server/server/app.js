const express = require('express');
const app = express();
const fs = require("fs");
const mongoose = require('mongoose')
const cors = require('cors');

// const MONGODB_URI = 'mongodb+srv://user1:admin@cluster0.aradevr.mongodb.net/?retryWrites=true&w=majority'

// mongoose.connect(MONGODB_URI, {
//     useNewUrlParser:true,
//     useUnifiedTopology:true,
// })

const questionString = fs.readFileSync('./questions.json');
const questionsArray = JSON.parse(questionString);

// const db = mongoose.connection;

// db.on('error', console.error.bind(console, 'connection error'));
// db.once('open', ()=>{
//     console.log('connected');
// })

app.use(cors());

app.post('/listUser/:id', function (req, res) {
   // First read existing users.

   console.log("hi");
   res.status(200).json({message: 'done'})
   
})

app.post('/addUser', function (req, res) {
    // First read existing users.
 
    console.log("hi");
     res.status(200).json({message: 'done'})
    
});

app.get('/random', (req, res) => {
    const randomQuestion = questionsArray[(Math.floor(Math.random() * questionsArray.length))].question;

    return res.json({
        "question": randomQuestion
    });
});

var server = app.listen(8081, function () {
   var host = server.address().address
   var port = server.address().port
   console.log("Server listening at http://localhost", host, port)
})