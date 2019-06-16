package redesneuraisadaline;

/*
Aluno: Yuri de Arruda Varão
Bloco: 6
*/

public class Adaline {

   //Declarando as variáveis que serão usadas
    private double u;
    private int epocas;
    private final double taxaAprendizado;
    private final double precisao;
    private double[] pesos = new double[5];
    private final double[][] matrizAprendizado;
    private final double[][] amostras;
    
    // Método construtor de inicialização os valores das variáveis e da matriz de aprendizado
    public Adaline() {
    
    // Amostras de Aprendizados da rede
    this.matrizAprendizado = new double[][]{
        {0.4329, -1.3719,  0.7022, -0.8535,  1.0000}, //Amostra 01
        {0.3024,  0.2286,  0.8630,  2.7909, -1.0000}, //Amostra 02 
        {0.1349, -0.6445,  1.0530,  0.5687, -1.0000}, //Amostra 03 
        {0.3374, -1.7163,  0.3670, -0.6283, -1.0000}, //Amostra 04 
        {1.1434, -0.0485,  0.6637,  1.2606,  1.0000}, //Amostra 05 
        {1.3749, -0.5071,  0.4464,  1.3009,  1.0000}, //Amostra 06 
        {0.7221, -0.7587,  0.7681, -0.5592,  1.0000}, //Amostra 07 
        {0.4403, -0.8072,  0.5154, -0.3129,  1.0000}, //Amostra 08 
        {-0.5231, 0.3548,  0.2538,  1.5776, -1.0000}, //Amostra 09 
        {0.3255, -2.0000,  0.7112, -1.1209,  1.0000}, //Amostra 10 
        {0.5824,  1.3915, -0.2291,  4.1735, -1.0000}, //Amostra 11
        {0.1340,  0.6081,  0.4450,  3.2230, -1.0000}, //Amostra 12 
        {0.1480, -0.2988,  0.4778,  0.8649,  1.0000}, //Amostra 13 
        {0.7359,  0.1869, -0.0872,  2.3584,  1.0000}, //Amostra 14 
        {0.7115, -1.1469,  0.3394,  0.9573, -1.0000}, //Amostra 15 
        {0.8251, -1.2840,  0.8452,  1.2382, -1.0000}, //Amostra 16 
        {0.1569,  0.3712,  0.8825,  1.7633,  1.0000}, //Amostra 17 
        {0.0033,  0.6835,  0.5389,  2.8249, -1.0000}, //Amostra 18 
        {0.4243,  0.8313,  0.2634,  3.5855, -1.0000}, //Amostra 19
        {1.0490,  0.1326,  0.9138,  1.9792,  1.0000}, //Amostra 20
        {1.4276,  0.5331, -0.0145,  3.7286,  1.0000}, //Amostra 21 
        {0.5971,  1.4865,  0.2904,  4.6069, -1.0000}, //Amostra 22
        {0.8475,  2.1479,  0.3179,  5.8235, -1.0000}, //Amostra 23 
        {1.3967, -0.4171,  0.6443,  1.3927,  1.0000}, //Amostra 24 
        {0.0044,  1.5378,  0.6099,  4.7755, -1.0000}, //Amostra 25 
        {0.2201, -0.5668,  0.0515,  0.7829,  1.0000}, //Amostra 26 
        {0.6300, -1.2480,  0.8591,  0.8093, -1.0000}, //Amostra 27 
        {-0.2479, 0.8960,  0.0547,  1.7381,  1.0000}, //Amostra 28 
        {-0.3088,-0.0929,  0.8659,  1.5483, -1.0000}, //Amostra 29 
        {-0.5180, 1.4974,  0.5453,  2.3993,  1.0000}, //Amostra 30
        {0.6833,  0.8266,  0.0829,  2.8864,  1.0000}, //Amostra 31 
        {0.4353, -1.4066,  0.4207, -0.4879,  1.0000}, //Amostra 32 
        {-0.1069,-3.2329,  0.1856, -2.4572, -1.0000}, //Amostra 33 
        {0.4662,  0.6261,  0.7304,  3.4370, -1.0000}, //Amostra 34 
        {0.8298, -1.4089,  0.3119,  1.3235, -1.0000}, //Amostra 35
    };
    
    // Amostras que serão classificadas pela rede
    this.amostras = new double[][] {
        {0.9694,  0.6909,  0.4334,  3.4965}, //Amostra 01 
        {0.5427,  1.3832,  0.6390,  4.0352}, //Amostra 02 
        {0.6081, -0.9196,  0.5925,  0.1016}, //Amostra 03 
        {-0.1618, 0.4694,  0.2030,  3.0117}, //Amostra 04 
        {0.1870, -0.2578,  0.6124,  1.7749}, //Amostra 05 
        {0.4891, -0.5276,  0.4378,  0.6439}, //Amostra 06 
        {0.3777,  2.0149,  0.7423,  3.3932}, //Amostra 07 
        {1.1498, -0.4067,  0.2469,  1.5866}, //Amostra 08 
        {0.9325,  1.0950,  1.0359,  3.3591}, //Amostra 09 
        {0.5060,  1.3317,  0.9222,  3.7174}, //Amostra 10 
        {0.0497, -2.0656,  0.6124, -0.6585}, //Amostra 11 
        {0.4004,  3.5369,  0.9766,  5.3532}, //Amostra 12 
        {-0.1874, 1.3343,  0.5374,  3.2189}, //Amostra 13 
        {0.5060,  1.3317,  0.9222,  3.7174}, //Amostra 14 
        {1.6375, -0.7911,  0.7537,  0.5515}  //Amostra 15 
    };
    
    // Valores aleatórios entre 0 e 1 para cada elemento do matriz peso.
    pesos[0] = Math.random();
    pesos[1] = Math.random();
    pesos[2] = Math.random();
    pesos[3] = Math.random();
    pesos[4] = Math.random();
    
    // Taxa de aprendizado para correção dos pesos
    this.taxaAprendizado = 0.0025;
    // Taxa de precisão requerida
    this.precisao = 0.000001;
    // Valor usado para representar o resultado da função de ativação
    this.u = 0;
    // Inicializa o número de Epocas a serem executadas para encontrar a solução
    this.epocas = 0;
    }
    
    // Método da Função de Ativação responsável pelo somatório das entradas multiplicadas com seus respectivos pesos.
    double funcaoAtivacao(double x1, double x2, double x3, double x4) {
        u = (x1 * pesos[0]) + (x2 * pesos[1]) + (x3 * pesos[2]) +  (x4 * pesos[3]) + ((-1) * pesos[4]);
        if (u >= 0) {
            return 1.0000;
        }
        return -1.0000;
    }
    
    // Método para a correção de pesos
    void corrigirPeso(int i, double u) {
        pesos[0] = pesos[0] + (taxaAprendizado * (matrizAprendizado[i][4] - u) * matrizAprendizado[i][0]);
        pesos[1] = pesos[1] + (taxaAprendizado * (matrizAprendizado[i][4] - u) * matrizAprendizado[i][1]);
        pesos[2] = pesos[2] + (taxaAprendizado * (matrizAprendizado[i][4] - u) * matrizAprendizado[i][2]);
        pesos[3] = pesos[3] + (taxaAprendizado * (matrizAprendizado[i][4] - u) * matrizAprendizado[i][3]);
        pesos[4] = pesos[4] + (taxaAprendizado * (matrizAprendizado[i][4] - u) * (-1));
    }
    
    // Método para encontrar o erro quadrático médio
    public double EQM(double[] pesos) {
        int padroesTreinamento = matrizAprendizado.length;
        double eqm = 0;
        for (int i = 0; i < padroesTreinamento; i++) {
            u = (matrizAprendizado[i][0] * pesos[0]) + (matrizAprendizado[i][1] * pesos[1]) + (matrizAprendizado[i][2] * pesos[2])
                + (matrizAprendizado[i][3] * pesos[3]) + ((-1) * pesos[4]);
            eqm = eqm + ((matrizAprendizado[i][4] - u) * (matrizAprendizado[i][4] - u));
        }
        return (eqm/padroesTreinamento);   
    }
    
    // Método que retorna o módulo do resultado de (eqmAtual - eqmAnterior)
    double modulo(double eqmAtual, double eqmAnterior){
        if ((eqmAtual - eqmAnterior) >= 0){
            return (eqmAtual - eqmAnterior);
        }else
            return ((eqmAtual - eqmAnterior)*(-1));
    } 
    
    // Método para treinamento da rede Adaline
    public void treinar() {
        //Variáveis locais de eqm usadas no método
        double eqmAnterior, eqmAtual;
        
        // Exibe a epoca em que ocorre cada iteração
        System.out.printf("Epoca: %d\n", epocas + 1);
        // Exibe os pesos utilizados em cada epoca
        System.out.printf("Pesos Utilizados:\nPeso 1: %.2f Peso 2: %.2f Peso 3: %.2f Peso 4: %.2f Peso 5: %.2f\n\n", 
            pesos[0], pesos[1], pesos[2], pesos[3], pesos[4]);
       
        // Encontra a eqmAnterior utilizando os pesos antes de corrigi-los
        eqmAnterior = EQM(pesos);
        
        // Laço usado para verificar todas as entradas e corrigir os pesos
        for (int i = 0; i < matrizAprendizado.length; i++) {
            funcaoAtivacao(matrizAprendizado[i][0], matrizAprendizado[i][1], matrizAprendizado[i][2], matrizAprendizado[i][3]);
            corrigirPeso(i, u);
        }
        this.epocas++;
        
        // Encontra a eqmAtual utilizando os pesos corrigidos
        eqmAtual = EQM(pesos);
        
        // Exibe as eqm's utilizadas, o módolo da diferença e a precisão requerida 
        System.out.printf("eqm anterior: %.6f eqm atual: %.6f\n", eqmAnterior, eqmAtual);
        System.out.printf("módulo-eqm: %.6f precisão:%.6f\n", modulo(eqmAtual, eqmAnterior), precisao);
        System.out.printf("\n==================================================\n\n");
        
        // Verifica se o treinamento ja terminou
        if(modulo(eqmAtual, eqmAnterior) <= precisao) {
            // Se o treinamento já ocorreu exibe msg
            System.out.printf("A rede foi treinada. Total de %d epocas.\n", epocas);
            System.out.printf("\n==================================================\n\n");
        }else{
            // Se ainda não ocorreu chama o método novamente
            treinar(); 
        }
    }
    
    // Método para classicar amostras com a rede treinada
    public void classificacao(){
        double sinal;
        System.out.printf("Classificação das amostras de operação:\n");
        for(int j = 0; j < amostras.length; j++){
            sinal = funcaoAtivacao(amostras[j][0], amostras[j][1], amostras[j][2], amostras[j][3]);
            if(sinal == -1.0000){
                System.out.printf("Amostra %d pertence a válvula A\n", j+1);
            }else if(sinal == 1.0000){
                System.out.printf("Amostra %d pertence a válvula B\n", j+1);
            }
        }
        System.out.printf("Classificação das amostras finalizada!\n");
        System.out.printf("\n==================================================\n\nFim do Programa!\n");
    }
    
    public static void main(String[] args) {
        //Instancia um objeto 
        Adaline adal = new Adaline();
        // Chama o método para Treinar a rede Adaline
        adal.treinar();
        // Chama o método para Classificar as amostras
        adal.classificacao();
    }  
}
