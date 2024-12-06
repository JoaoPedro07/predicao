from django.shortcuts import render
from .preditor import PrevisaoDengue
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse, inline_serializer
from rest_framework import status
from dataclasses import asdict
from typing import Dict
import json
# Create your views here.

class PredicaoDoenca(APIView):
    @extend_schema(
    parameters=[
            OpenApiParameter('geocodigo', type=int, location='query', description='Geocodigo(disponível na base de dados como referência)'),
            OpenApiParameter('doenca', type=str, location='query', description='Doença(dengue, zika, chikungunya)'),
            OpenApiParameter('ano_inicio', type=int, location='query', description='Ano de início(ex: 2015)'),
            OpenApiParameter('ano_fim', type=int, location='query', description='Ano de fim(ex: 2024)'),
        ],
    )
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
    
