// Auth.ts
import * as AWSCognitoIdentity from 'amazon-cognito-identity-js'

/*
* User Pool ID와 Client ID를 겍체를 생성
* 이 객체는 CognitoUserPool 객체를 생성할 때 사용
* */
const userPoolData: AWSCognitoIdentity.ICognitoUserPoolData = {
    UserPoolId: 'ap-northeast-2_XXXXXXXXX',
    ClientId: '78q2qbXXXXXXXXXXXXXXXXXXXX'
}


/*
* 사용자가 회원 가입을하면 해당 이메일로 Confirmation code가 발송됨
* */
export async function confirm({ Username, ConfirmationCode }: { Username: string, ConfirmationCode: string }): Promise<any> {
    const userData: AWSCognitoIdentity.ICognitoUserData = {
        Username: Username,
        Pool: new AWSCognitoIdentity.CognitoUserPool(userPoolData)
    }

    const cognitoUser: AWSCognitoIdentity.CognitoUser = new AWSCognitoIdentity.CognitoUser(userData)
    /*
    * CognitoUser.confirmRegistration() 함수에 Confirmation code 전달 
    * */
    return await new Promise((resolve, reject) => {
        cognitoUser.confirmRegistration(ConfirmationCode, true, (err, result) => {
            if(err)
                reject(err.message || JSON.stringify(err))
            else
                resolve(result)
        })
    })
}