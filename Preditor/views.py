from django.shortcuts import render
from .preditor import PrevisaoDengue
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from dataclasses import asdict
from typing import Dict
import json
# Create your views here.

class PredicaoDoenca(APIView):
    def post(self, request):
        try:
            geocodigo = request.data.get('geocodigo')
            doenca = request.data.get('doenca')
            ano_inicio = request.data.get('ano_inicio')
            ano_fim = request.data.get('ano_fim')

            previsor = PrevisaoDengue()
            resultado = previsor.prever(
                geocodigo=geocodigo,
                doenca=doenca,
                ano_inicio=ano_inicio,
                ano_fim=ano_fim
            )
            dados_dict = asdict(resultado)
            dados_dict['dados_historicos'] = resultado.dados_historicos.to_dict(orient='records')
            dados_dict['previsoes'] = resultado.previsoes.to_dict(orient='records')
            formatado = dados_dict 
            return Response({
                    'prediction': formatado,
                    'message': 'Modelo treinado e previsão realizada com sucesso.'
                }, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
