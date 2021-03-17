from django.http import JsonResponse

class Response:

	def base(delf, data=None, message="", error=False, status=200):

		if data is None:
			data = []

		return JsonResponse({
			'data': data,
			'message': message,
			'error': error
		}, status=status)

	@staticmethod
	def ok(data=None, message="", status=200):
		return Response().base(data=data, message=message, error=False, status=status)

	@staticmethod
	def badRequest(data=None, message="", status=400):
		return Response().base(data=data, message=message, error=True, status=status)