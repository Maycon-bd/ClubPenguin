# 🌾 Especificações de Parâmetros Morfológicos para Identificação de Grãos

Este guia descreve os parâmetros físicos e geométricos utilizados pelo modelo de Machine Learning do **MatoScan** para discriminar e identificar sementes das cinco culturas de grãos suportadas (Milho, Soja, Arroz, Trigo e Feijão).

---

## 📐 1. Descrição dos Parâmetros Morfológicos

O sistema utiliza duas categorias de parâmetros para traçar o perfil biométrico de cada semente: **Medidas Primárias** (coletadas diretamente) e **Medidas Derivadas** (calculadas geometricamente a partir das primárias).

### Medidas Primárias
1. **Comprimento (mm)**: O comprimento do eixo longitudinal maior da semente. Reflete o crescimento e a extensão celular principal.
2. **Largura (mm)**: O diâmetro do eixo transversal menor. Representa a espessura lateral da semente na projeção horizontal.
3. **Espessura (mm)**: A profundidade tridimensional perpendicular ao plano de comprimento-largura.
4. **Massa (mg)**: Peso individual de uma semente desidratada comercialmente. Relacionado à densidade de endosperma e acúmulo de reservas de nutrientes.

### Medidas Derivadas
5. **Área Projetada (mm²)**: Área estimada da elipse formada pelo comprimento ($L$) e pela largura ($W$):
   $$Area = \frac{\pi \cdot L \cdot W}{4}$$
6. **Perímetro (mm)**: Perímetro estimado da elipse utilizando a segunda aproximação de Ramanujan para alta precisão:
   $$Perimetro \approx \pi \left[ 3(a + b) - \sqrt{(3a + b)(a + 3b)} \right] \quad \text{onde } a = \frac{L}{2}, \, b = \frac{W}{2}$$
7. **Compacidade (Razão de Circularidade)**: Mede o quão circular é a projeção da semente. Uma compacidade próxima a $1.0$ representa um círculo perfeito:
   $$Compacidade = \frac{4\pi \cdot Area}{Perimetro^2}$$
8. **Razão de Aspecto (Alongamento)**: A relação entre o maior e o menor eixo:
   $$Razao\_Aspecto = \frac{Comprimento}{Largura}$$

---

## 🌾 2. Fichas Morfológicas por Cultura

Abaixo estão detalhados os limites estatísticos (médias e desvios padrão) de cada cultura simulados realisticamente no gerador de dados do **MatoScan**:

### 🌽 Milho (*Zea mays*)
As sementes de milho apresentam formato achatado e cônico, com base estreita e topo largo (geralmente dentado).
* **Comprimento médio**: $10.0$ mm (faixa: $8.0$ a $12.0$ mm)
* **Largura média**: $8.0$ mm (faixa: $6.0$ a $10.0$ mm)
* **Espessura média**: $5.0$ mm (faixa: $4.0$ a $6.0$ mm)
* **Massa média**: $320$ mg (faixa: $250$ a $400$ mg)
* **Razão de Aspecto típica**: $1.25$ (forma levemente alongada/trapezoidal)
* **Compacidade típica**: $0.94$ a $0.97$

### 🫘 Soja (*Glycine max*)
A semente da soja destaca-se por sua geometria altamente esférica (globosa), com hilos discretos e superfície lisa.
* **Comprimento médio**: $6.5$ mm (faixa: $5.0$ a $8.0$ mm)
* **Largura média**: $6.0$ mm (faixa: $5.0$ a $7.0$ mm)
* **Espessura média**: $5.0$ mm (faixa: $4.0$ a $6.0$ mm)
* **Massa média**: $195$ mg (faixa: $150$ a $250$ mg)
* **Razão de Aspecto típica**: $1.08$ (quase esférica)
* **Compacidade típica**: $0.98$ a $1.00$ (a maior compacidade entre as culturas)

### 🌾 Arroz (*Oryza sativa*)
As sementes de arroz (com casca/parietal) são lineares, extremamente alongadas, estreitas e achatadas lateralmente.
* **Comprimento médio**: $7.5$ mm (faixa: $6.0$ a $9.0$ mm)
* **Largura média**: $2.4$ mm (faixa: $2.0$ a $3.0$ mm)
* **Espessura média**: $2.0$ mm (faixa: $1.5$ a $2.5$ mm)
* **Massa média**: $26$ mg (faixa: $18$ a $35$ mg)
* **Razão de Aspecto típica**: $3.12$ (muito alongada, grão agulha)
* **Compacidade típica**: $0.60$ a $0.75$ (a menor compacidade do grupo)

### 🌾 Trigo (*Triticum aestivum*)
Os grãos de trigo são elípticos, com sulco longitudinal característico e pequenas penugens na extremidade apical. São menores que o feijão e mais largos que o arroz.
* **Comprimento médio**: $6.0$ mm (faixa: $5.0$ a $7.0$ mm)
* **Largura média**: $3.5$ mm (faixa: $3.0$ a $4.0$ mm)
* **Espessura média**: $3.0$ mm (faixa: $2.5$ a $3.5$ mm)
* **Massa média**: $42$ mg (faixa: $30$ a $55$ mg)
* **Razão de Aspecto típica**: $1.71$ (intermediária)
* **Compacidade típica**: $0.88$ a $0.92$

### 🫘 Feijão (*Phaseolus vulgaris*)
O grão de feijão possui formato reniforme (em formato de rim) bem acentuado, sendo a maior semente do grupo em volume e comprimento absoluto.
* **Comprimento médio**: $15.0$ mm (faixa: $12.0$ a $18.0$ mm)
* **Largura média**: $9.0$ mm (faixa: $7.0$ a $11.0$ mm)
* **Espessura média**: $7.0$ mm (faixa: $6.0$ a $8.0$ mm)
* **Massa média**: $320$ mg (faixa: $200$ a $450$ mg)
* **Razão de Aspecto típica**: $1.67$ (alongamento médio)
* **Compacidade típica**: $0.85$ a $0.90$

---

## 📊 3. Tabela Comparativa de Referência (Médias)

Esta tabela consolida os valores médios de cada grão para facilitar o ajuste rápido nos seletores do sistema:

| Cultura | Comprimento (mm) | Largura (mm) | Espessura (mm) | Massa (mg) | Razão de Aspecto | Compacidade |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **🌽 Milho** | $10.0$ | $8.0$ | $5.0$ | $320.0$ | $1.25$ | $0.96$ |
| **🫘 Soja** | $6.5$ | $6.0$ | $5.0$ | $195.0$ | $1.08$ | $0.99$ |
| **🌾 Arroz** | $7.5$ | $2.4$ | $2.0$ | $26.0$ | $3.12$ | $0.67$ |
| **🌾 Trigo** | $6.0$ | $3.5$ | $3.0$ | $42.0$ | $1.71$ | $0.91$ |
| **🫘 Feijão** | $15.0$ | $9.0$ | $7.0$ | $320.0$ | $1.67$ | $0.88$ |
