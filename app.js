// Required dependencies
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const admin = require('firebase-admin');

// Firebase setup
const serviceAccount = require('D:/clone_ai/ai-clone-47beb-firebase-adminsdk-nrs6e-f5d7911fac.json'); // Replace with your Firebase service account key file
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: 'https://ai-clone-47beb-default-rtdb.firebaseio.com'
});

const db = admin.database();

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Serve static files
app.use(express.static('public'));

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

// API for handling file upload
app.post('/upload', upload.single('file'), (req, res) => {
    if (req.file) {
        console.log('File uploaded:', req.file.filename);

        // Extract and prepare data for training
        const uploadedFilePath = path.join(__dirname, req.file.path);
        const userData = {
            name: req.body.name || "Avatar",
            tweets: fs.readFileSync(uploadedFilePath, 'utf-8').split('\n').filter((line) => line.trim()),
        };

        // Save user data to Firebase
        const userId = saveToFirebase('userResponses', userData);

        // Train the model using uploaded file data
        triggerModelTraining(userData, null, userId);

        res.status(200).send({ message: 'File uploaded successfully and training started!' });
    } else {
        res.status(400).send({ message: 'File upload failed.' });
    }
});

// Questions for the chatbot
const personalityTestQuestions = [
    { question: "Describe yourself in one sentence.", type: "text" },
    { question: "What tone best represents your style?", type: "options", options: ["Friendly", "Formal", "Inspirational", "Conversational", "Humorous", "Witty", "Assertive"] },
    { question: "What topics or themes do you usually write about?", type: "text" },
    { question: "Provide a sample of your writing (Example 1).", type: "text" },
    { question: "Provide a sample of your writing (Example 2).", type: "text" },
    { question: "Provide a sample of your writing (Example 3).", type: "text" },
    { question: "Are there any words, topics, or tones you'd like the AI to avoid?", type: "text" },
    { question: "If you use humor, how would you describe it?", type: "text" },
    { question: "Do you have any punctuation or formatting quirks you use often?", type: "text" },
    { question: "Who is your typical audience?", type: "text" },
    { question: "What’s one thing you want your AI clone to always reflect about you?", type: "text" },
];

// Sanitize keys for Firebase
const sanitizeKey = (key) => key.replace(/[.$#[\]/]/g, '_');

// Save data to Firebase
const saveToFirebase = (key, data) => {
    const sanitizedData = {};
    for (const k in data) {
        const sanitizedKey = sanitizeKey(k);
        sanitizedData[sanitizedKey] = data[k];
    }

    const ref = db.ref(key);
    const newRef = ref.push(sanitizedData, (err) => {
        if (err) {
            console.error('Error saving to Firebase:', err);
        } else {
            console.log('Data saved to Firebase');
        }
    });
    return newRef.key; // Return the unique ID of the new entry
};

// Temporary storage for user responses
const userResponses = {};

// Handle socket connection
io.on('connection', (socket) => {
    console.log('A user connected');
    let currentQuestion = 0;
    let flow = "name"; // Tracks the flow: 'name', 'choice', 'test', 'upload'
    let isTrained = false; // Tracks if the model is trained

    const sendQuestion = () => {
        if (flow === "name") {
            socket.emit('chatMessage', {
                sender: 'bot',
                message: "What's your name?"
            });
        } else if (flow === "choice") {
            socket.emit('chatMessage', {
                sender: 'bot',
                message: "How would you like to proceed?",
                options: ["Take Personality Test", "Upload Twitter Archive"]
            });
        } else if (flow === "test") {
            if (currentQuestion < personalityTestQuestions.length) {
                const question = personalityTestQuestions[currentQuestion];
                if (question.type === "options") {
                    socket.emit('chatMessage', {
                        sender: 'bot',
                        message: question.question,
                        options: question.options
                    });
                } else {
                    socket.emit('chatMessage', {
                        sender: 'bot',
                        message: question.question
                    });
                }
            } else {
                socket.emit('chatMessage', {
                    sender: 'bot',
                    message: `Thank you, ${userResponses.name}! Your responses are being used to train your AI clone. Please wait...`
                });
                const userId = saveToFirebase('userResponses', userResponses);
                triggerModelTraining(userResponses, socket, userId);
            }
        } else if (flow === "upload") {
            socket.emit('chatMessage', {
                sender: 'bot',
                message: "Please upload your Twitter archive file for better personalization. This file helps us analyze your style more accurately. We’re also working on integrating the Twitter API for enhanced features in the future.",
                fileUpload: true
            });
        }
    };

    const triggerModelTraining = (responses, socket, userId) => {
        const dataPath = path.join(__dirname, 'user_data.json');
        fs.writeFileSync(dataPath, JSON.stringify(responses, null, 2));

        // Call the Python script to train the model
        const scriptPath = path.join(__dirname, 'train_model.py');
        exec(`python ${scriptPath} ${dataPath}`, (error, stdout, stderr) => {
            if (error) {
                console.error(`Training error: ${stderr}`);
                if (socket) socket.emit('chatMessage', { sender: 'bot', message: "Training failed. Please try again later." });
                return;
            }
            console.log(`Training output: ${stdout}`);
            const modelPath = `./models/${responses.name.replace(' ', '_').toLowerCase()}_avatar`;
            const ref = db.ref(`userResponses/${userId}`);
            ref.update({ modelPath: modelPath }, (err) => {
                if (err) {
                    console.error('Error updating modelPath in Firebase:', err);
                } else {
                    console.log('Model path updated successfully in Firebase');
                }
            });
            isTrained = true;
            if (socket) {
                socket.emit('chatMessage', { sender: 'bot', message: `Training complete! Your avatar's name is ${responses.name}. 🎉` });
                socket.emit('chatMessage', { sender: 'bot', message: "Your AI clone is ready to use! Ask it to write a tweet or anything you'd like." });
            }
        });
    };

    const generateResponse = (userMessage, socket) => {
        if (!isTrained) {
            socket.emit('chatMessage', { sender: 'bot', message: "Your AI clone is still being trained. Please wait..." });
            return;
        }

        // Call a Python script to generate a response using the trained model
        const scriptPath = path.join(__dirname, 'generate_response.py');
        exec(`python ${scriptPath} "${userMessage}"`, (error, stdout, stderr) => {
            if (error) {
                console.error(`Response generation error: ${stderr}`);
                socket.emit('chatMessage', { sender: 'bot', message: "Sorry, I couldn't generate a response. Please try again." });
                return;
            }
            console.log(`Generated response: ${stdout}`);
            socket.emit('chatMessage', { sender: 'bot', message: stdout });
        });
    };

    socket.on('chatMessage', (data) => {
        if (flow === "name") {
            userResponses["name"] = data.message;
            flow = "choice";
            sendQuestion();
        } else if (flow === "choice") {
            if (data.message === "Take Personality Test") {
                flow = "test";
                currentQuestion = 0;
                sendQuestion();
            } else if (data.message === "Upload Twitter Archive") {
                flow = "upload";
                sendQuestion();
            }
        } else if (flow === "test") {
            // Ensure currentQuestion is within bounds
            if (currentQuestion < personalityTestQuestions.length) {
                userResponses[personalityTestQuestions[currentQuestion].question] = data.message;
                currentQuestion++;
                sendQuestion();
            } else {
                socket.emit('chatMessage', {
                    sender: 'bot',
                    message: `Thank you, ${userResponses.name}! Your responses are being used to train your AI clone. Please wait...`
                });
                const userId = saveToFirebase('userResponses', userResponses);
                triggerModelTraining(userResponses, socket, userId);
            }
        } else if (flow === "upload") {
            socket.emit('chatMessage', {
                sender: 'bot',
                message: "Click the button below to upload your file.",
                fileUpload: true
            });
        } else {
            generateResponse(data.message, socket);
        }
    });

    socket.on('disconnect', () => {
        console.log('A user disconnected');
    });

    sendQuestion();
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
