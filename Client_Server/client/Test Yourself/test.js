const recognition = new webkitSpeechRecognition || new SpeechRecognition();
recognition.continuous = true;
recognition.interimResults = false;

const URL = "http://localhost:8081/";

const textTranscript = document.querySelector("#text-transcript");
const btnRecordAnswer = document.querySelector("#btn-record-answer");
const btnGetQuestion = document.querySelector("#btn-random-question");
const btnGetFeedback = document.querySelector("#btn-get-feedback");
const questionContainer = document.querySelector("#question");
const wordsLength = document.querySelector("#li-words-length");
const timeSeconds = document.querySelector("#li-seconds");
const wpm = document.querySelector("#li-wpm");

let isRecording = false;

let second = 0;
let count = 0;
let timer = false;

const ctx = document.getElementById('myChart');
ctx.style.display = "none";

const updateWPM = (t, w) => (w / t) * 60

new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Confidence', 'Joy', 'Determination', 'Anger'],
        datasets: [{
            data: [10, 9, 5, 0],
            borderWidth: 1,
            backgroundColor: "#5B0888"
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }, plugins: {
            legend: {
                labels: {
                    // This more specific font property overrides the global property
                    font: {
                        size: 16,
                        family: 'Inter',
                        weight: 500
                    }
                }
            }
        }
    }
});

function countWords(str) {
    const arr = str.split(' ');

    return arr.filter(word => word !== '').length;
}

textTranscript.oninput = () => {
    wordsLength.innerHTML = `Length: ${countWords(textTranscript.value)} words`;
}

btnGetFeedback.addEventListener('click', () => {
    ctx.style.display = "block";
})

btnRecordAnswer.addEventListener('click', () => {
    if (isRecording)
        recognition.stop();
    else
        recognition.start();

    isRecording = !isRecording;
});

btnGetQuestion.addEventListener('click', () => {
    btnGetQuestion.classList.add("button--loading");
    btnGetQuestion.disabled = true;
    btnGetQuestion.innerHTML = "";

    fetch(URL + "random", {
        method: "GET",
    })
        .then(res => res.json())
        .then(json => {
            btnGetQuestion.classList.remove("button--loading");
            btnGetQuestion.disabled = false;
            btnGetQuestion.innerHTML = "Get Random Question";

            questionContainer.innerHTML = `Question: ${json.question}`;
        });
});



recognition.onstart = () => {
    btnRecordAnswer.innerHTML = "Stop Recording";
    btnRecordAnswer.style.backgroundColor = "#4CAF50";
    timer = true;
    second = 0;
    count = 0;
    stopWatch();
}

recognition.onend = () => {
    btnRecordAnswer.innerHTML = "Record Answer";
    btnRecordAnswer.style.backgroundColor = "#5B0888";
    timer = false;
    timeSeconds.innerHTML = `Time: ${second}.${count} seconds`;
}

recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    textTranscript.value = transcript;
    wpm.innerHTML = `WPM: ${updateWPM(second, countWords(transcript))} WPM`;
    wordsLength.innerHTML = `Length: ${countWords(transcript)} words`;
}

function stopWatch() {
    if (timer) {
        count++;

        if (count === 100) {
            second++;
            count = 0;
        }

        setTimeout(stopWatch, 10);
    }
}