console.log("res");

document.getElementById("Submit").addEventListener("click",async(e)=>{
    e.preventDefault();
    console.log("herel,o");
    const FullName=document.getElementById("FullName");
    const EmailId=document.getElementById("EmailId");
    const CreatePassword=document.getElementById("CreatePassword");
    const bFN=FullName.value;
    const bEI=EmailId.value;
    const bCP=CreatePassword.value;
    const body={
        FullName:bFN,
        EmailId:bEI,
        CreatePassword:bCP
    };
    FullName.value="";
    EmailId.value="";
    CreatePassword.value="";
    const ans=await fetch("https//localhost:3000",
                        {
                        method:"post",
                        headers: {
                            "Content-Type": "application/json",
                          },
                        body:JSON.stringify(body)
                    }
    );
    const res=await ans.json();
    console.log(res);
})