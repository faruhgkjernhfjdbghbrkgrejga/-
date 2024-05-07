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


export async function signUp({ Username, Password, Email }: { Username: string, Password: string, Email: string }): Promise<{ message: string }> {
    /*
    * Required attributes를 추가
    * */
    const attributeData: AWSCognitoIdentity.ICognitoUserAttributeData = {
        Name: 'email',
        Value: Email
    }

    let attributeList: AWSCognitoIdentity.CognitoUserAttribute[] = [
        new AWSCognitoIdentity.CognitoUserAttribute(attributeData)
    ]
    /*
    * CognitoUserPool.signUp() 함수에 다음과 같이 Username, Password, Required attributes를 전달
    * 콜백함수를 통해 결과를 반환
    * */
    return await new Promise((resolve, reject) => {
        const userPool = new AWSCognitoIdentity.CognitoUserPool(userPoolData)

        userPool.signUp(Username, Password, attributeList, attributeList,
            (err: Error | undefined, result: AWSCognitoIdentity.ISignUpResult | undefined): void => {

                if(err)
                    reject({ message: err.message || JSON.stringify(err) })
                else
                    resolve({ message: result?.user.getUsername() + '님, 회원 가입이 성공적으로 완료되었습니다.' })

            })
    })