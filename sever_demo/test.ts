// test.ts
import { signUp } from "./Auth";

(async function (){
    let resultMessage = await signUp({
        Username: 'user1234',
        Password: 'Abcde12345**',
        Email: 'example@example.com'
    }).catch(console.log)

    console.log(resultMessage)
})()