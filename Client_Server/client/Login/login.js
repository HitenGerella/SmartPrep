console.log("res");

document.getElementById("Submit").addEventListener("click",async(e)=>{
    e.preventDefault();
    console.log("herel,o");
    // const FullName=document.getElementById("FullName");
    const EmailId=document.getElementById("EmailId");
    const Password=document.getElementById("Password");
    // const bFN=FullName.value;
    const bEI=EmailId.value;
    const bP=Password.value;
    const body={
        // FullName:bFN,
        EmailId:bEI,
        Password:bP
    };
    // FullName.value="";
    EmailId.value="";
    Password.value="";
    const ans=await fetch('https://jsonplaceholder.typicode.com/posts', {
    method: 'POST',
    body: JSON.stringify(body),
    headers: {
        'Content-type': 'application/json; charset=UTF-8',
    },
    });
    const res=await ans.json();
    console.log(res);
//   .then((response) => response.json())
//   .then((json) => console.log(json));
})